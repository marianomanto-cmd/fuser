"""Orquestación del face swap de vídeo.

Pipeline de tres etapas que solapa I/O con cómputo de GPU usando la RAM como
buffer elástico (de ahí el provecho de los 40 GB):

    [hilo decodificador] --in_queue--> [GPU: este hilo] --out_queue--> [hilo escritor]
        (lee frames)                    (detecta+swap+realce)            (codifica H.264)

- Las colas tienen tamaño acotado (``prefetch_frames`` / ``writer_queue``) según
  el modo de memoria: dan *backpressure* para no llenar la RAM sin control.
- Todos los modelos se invocan desde un único hilo (este), evitando problemas de
  hilos con onnxruntime/insightface.
- El audio original se re-multiplexa al final.

Expone tres operaciones de alto nivel: ``load_models``, ``preview`` y
``process_video``.
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .. import config
from ..config import OUTPUTS_DIR, TEMP_DIR, Settings, ensure_dirs
from ..models.downloader import ensure_model
from ..utils import image as imageutil
from ..utils import video as videoutil
from ..utils.logging import get_logger
from .face_store import FaceStore
from .memory_manager import MemoryManager
from .temporal import TemporalSmoother

log = get_logger(__name__)

ProgressCb = Optional[Callable[[float, str], None]]


def format_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class SwapPipeline:
    """Pipeline completo de face swap, configurado por ``Settings``."""

    def __init__(self, settings: Settings):
        self.settings = settings.resolved()
        self.mm = MemoryManager(self.settings)
        self.analyser = None
        self.swapper = None
        self.enhancer = None
        self.enhancer_info = None
        self.store: Optional[FaceStore] = None
        self.smoother: Optional[TemporalSmoother] = None
        self._loaded = False

    # ----- Carga de modelos ----------------------------------------------------
    def load_models(self, progress: ProgressCb = None) -> None:
        if self._loaded:
            return
        from ..models.face_analyser import FaceAnalyser
        from ..models.face_enhancer import FaceEnhancer
        from ..models.face_swapper import FaceSwapper

        s = self.settings
        ensure_dirs()

        if progress:
            progress(0.05, "Cargando detector de caras…")
        self.analyser = FaceAnalyser(
            self.mm.analyser_providers(), self.mm.ctx_id(), self.mm.det_size
        )
        self.analyser.load()

        if progress:
            progress(0.25, "Preparando el modelo de swap…")
        swap_path = ensure_model(s.swapper_model, progress)
        self.swapper = FaceSwapper(swap_path, self.mm.swapper_providers())
        self.swapper.load()

        if s.enhancer_model and s.enhancer_model != "none":
            if progress:
                progress(0.55, "Preparando el enhancer…")
            info = config.ENHANCER_MODELS[s.enhancer_model]
            enh_path = ensure_model(s.enhancer_model, progress)
            on_cpu = (not self.mm.use_gpu) or s.preset.get("enhancer_device") == "cpu"
            self.enhancer = FaceEnhancer(
                enh_path, self.mm.enhancer_providers(), info, self.mm.session_options(on_cpu)
            )
            self.enhancer.load()
            self.enhancer_info = info

        self.store = FaceStore(self.analyser, s)
        self.smoother = TemporalSmoother(s.temporal_alpha) if s.temporal_smoothing else None
        self._loaded = True
        if progress:
            progress(1.0, "Modelos listos.")
        log.info("Pipeline listo · %s", self.mm.summary())

    @property
    def loaded(self) -> bool:
        return self._loaded

    def model_signature(self) -> tuple:
        """Identidad de los ajustes que obligan a recargar modelos."""
        s = self.settings
        return (
            s.swapper_model, s.enhancer_model, s.memory_mode,
            s.force_cpu, round(s.gpu_mem_limit_gb, 2), s.det_size,
        )

    def update_runtime(self, settings: Settings) -> None:
        """Aplica ajustes ligeros (máscara, opacidad, selector...) sin recargar.

        Solo debe llamarse cuando ``model_signature`` no cambia. Los parámetros
        de modelo/memoria requieren reconstruir el pipeline.
        """
        self.settings = settings.resolved()
        if self.store is not None:
            self.store.settings = self.settings
        if self.settings.temporal_smoothing:
            self.smoother = TemporalSmoother(self.settings.temporal_alpha)
        else:
            self.smoother = None

    # ----- Fuente / referencia -------------------------------------------------
    def prepare_source(self, images: List[np.ndarray]) -> None:
        if not self._loaded:
            self.load_models()
        self.store.set_source(images)

    def set_reference_from_frame(self, frame: np.ndarray, face_index: int = 0) -> bool:
        if not self._loaded:
            self.load_models()
        return self.store.set_reference(frame, face_index)

    # ----- Procesamiento de un frame ------------------------------------------
    def _output_dims(self, w: int, h: int) -> Tuple[int, int]:
        mx = self.settings.processing_resolution
        if mx <= 0 or max(w, h) <= mx:
            return w, h
        f = mx / float(max(w, h))
        return int(round(w * f)), int(round(h * f))

    def _aligned_target(self, frame: np.ndarray, target_face, size: int) -> np.ndarray:
        from insightface.utils import face_align

        return face_align.norm_crop(frame, target_face.kps, size)

    def _swap_one(self, frame: np.ndarray, target_face) -> np.ndarray:
        s = self.settings
        bgr_fake, affine = self.swapper.swap_raw(frame, target_face, self.store.source_face)
        face = bgr_fake

        if self.enhancer is not None:
            up = cv2.resize(bgr_fake, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            enhanced = self.enhancer.run(up, fidelity=s.codeformer_fidelity)
            blend = float(np.clip(s.enhancer_blend, 0.0, 1.0))
            face = cv2.addWeighted(enhanced, blend, up, 1.0 - blend, 0)
            affine = imageutil.scale_affine(affine, 512.0 / max(1, self.swapper.input_size))

        if s.color_match:
            ref = self._aligned_target(frame, target_face, face.shape[0])
            face = imageutil.apply_color_transfer(face, ref)

        return imageutil.paste_back(
            frame, face, affine,
            mask_blur=s.mask_blur, mask_padding=s.mask_padding, opacity=s.face_opacity,
        )

    def _process_frame(self, frame: np.ndarray, use_smoothing: bool = True) -> np.ndarray:
        work, _ = imageutil.limit_resolution(frame, self.settings.processing_resolution)
        faces = self.analyser.get_faces(work)
        if use_smoothing and self.smoother is not None:
            faces = self.smoother.smooth(faces, work.shape)
        targets = self.store.select_targets(faces)
        for tf in targets:
            work = self._swap_one(work, tf)
        return work

    # ----- Previsualización ----------------------------------------------------
    def preview(
        self, video_path: str, n_frames: int = 6, progress: ProgressCb = None
    ) -> List[Tuple[np.ndarray, str]]:
        if not self._loaded:
            self.load_models()
        if self.store is None or self.store.source_face is None:
            raise RuntimeError("Sube primero una imagen fuente con una cara visible.")

        info = videoutil.probe(video_path)
        indices = videoutil.keyframe_indices(info.frame_count, n_frames)
        frames = videoutil.get_frames_at(video_path, indices)

        results: List[Tuple[np.ndarray, str]] = []
        for i, (idx, frame) in enumerate(zip(indices, frames)):
            if progress:
                progress((i + 1) / max(1, len(frames)), f"Previsualizando frame {idx}…")
            out = self._process_frame(frame, use_smoothing=False)  # frames no consecutivos
            results.append((imageutil.to_rgb(out), f"Frame {idx}"))
        return results

    # ----- Procesamiento completo del vídeo -----------------------------------
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress: ProgressCb = None,
    ) -> str:
        if not self._loaded:
            self.load_models()
        if self.store is None or self.store.source_face is None:
            raise RuntimeError("Falta la cara fuente. Sube una imagen fuente primero.")

        ensure_dirs()
        info = videoutil.probe(video_path)
        out_w, out_h = self._output_dims(info.width, info.height)
        stamp = int(time.time())
        tmp_video = TEMP_DIR / f"fuser_tmp_{stamp}.mp4"
        final_out = Path(output_path) if output_path else OUTPUTS_DIR / f"fuser_{stamp}.mp4"

        writer = videoutil.FFmpegVideoWriter(
            str(tmp_video), out_w, out_h, info.fps,
            crf=self.settings.output_quality, encoder=self.settings.output_video_encoder,
        )

        in_q: "queue.Queue" = queue.Queue(maxsize=self.mm.prefetch_frames)
        out_q: "queue.Queue" = queue.Queue(maxsize=self.mm.writer_queue)
        errors: dict = {}

        def decoder():
            try:
                for frame in videoutil.read_frames(video_path):
                    in_q.put(frame)
            except Exception as exc:  # pragma: no cover
                errors["decode"] = exc
            finally:
                in_q.put(None)

        def writer_thread():
            try:
                while True:
                    item = out_q.get()
                    if item is None:
                        break
                    writer.write(item)
            except Exception as exc:  # pragma: no cover
                errors["write"] = exc

        dt = threading.Thread(target=decoder, daemon=True)
        wt = threading.Thread(target=writer_thread, daemon=True)
        dt.start()
        wt.start()

        total = max(1, info.frame_count)
        processed = 0
        start = time.time()
        if self.smoother:
            self.smoother.reset()

        try:
            while True:
                frame = in_q.get()
                if frame is None:
                    break
                if "decode" in errors:
                    raise errors["decode"]
                result = self._process_frame(frame, use_smoothing=True)
                out_q.put(result)
                processed += 1

                if progress:
                    elapsed = time.time() - start
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    remaining = (total - processed) / rate if rate > 0 else 0.0
                    frac = min(processed / total, 0.99)
                    progress(
                        frac,
                        f"Frame {processed}/{total} · {rate:.1f} fps · ETA {format_eta(remaining)}",
                    )
        finally:
            out_q.put(None)
            wt.join()
            writer.close()
            dt.join(timeout=2)

        if "decode" in errors:
            raise errors["decode"]
        if "write" in errors:
            raise errors["write"]

        # Re-multiplexado del audio original.
        if self.settings.keep_audio and info.has_audio:
            if progress:
                progress(0.997, "Incrustando audio…")
            if videoutil.mux_audio(str(tmp_video), video_path, str(final_out)):
                tmp_video.unlink(missing_ok=True)
            else:
                shutil.move(str(tmp_video), str(final_out))
        else:
            shutil.move(str(tmp_video), str(final_out))

        if progress:
            progress(1.0, "¡Vídeo listo!")
        log.info("Vídeo generado: %s (%d frames)", final_out, processed)
        return str(final_out)
