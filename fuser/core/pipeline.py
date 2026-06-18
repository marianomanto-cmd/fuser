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
from .temporal import TemporalSmoother, apply_two_pass_smoothing

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
        self.parser = None          # face parser (lazy, opcional)
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
        self.smoother = (
            TemporalSmoother(s.temporal_alpha, motion_adaptive=s.motion_adaptive)
            if s.temporal_smoothing else None
        )
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
            self.smoother = TemporalSmoother(
                self.settings.temporal_alpha, motion_adaptive=self.settings.motion_adaptive
            )
        else:
            self.smoother = None

    # ----- Fuente / referencia -------------------------------------------------
    def prepare_source(self, images: List[np.ndarray]):
        if not self._loaded:
            self.load_models()
        return self.store.set_source(images)

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

    @staticmethod
    def _aligned_crop(frame: np.ndarray, affine, size: int) -> np.ndarray:
        """Recorte alineado del frame usando la matriz afín (equivale a norm_crop)."""
        return cv2.warpAffine(frame, affine, (size, size), borderMode=cv2.BORDER_REPLICATE)

    def _mouth_open_boost(self) -> float:
        """Cuánto alargar la región de boca (mayor en modos de canto/expresión)."""
        m = self.settings.expression_mode
        if m == config.EXPR_HIGH_EXPRESSION:
            return 2.0
        if m == config.EXPR_MUSIC_VIDEO:
            return 1.5
        return 1.0

    def _ensure_parser(self):
        """Carga perezosa del face parser (solo si se usa la máscara 'parsing')."""
        if self.parser is not None:
            return self.parser
        from ..models.face_parser import FaceParser

        info = config.PARSER_MODELS["face_parser_bisenet"]
        path = ensure_model("face_parser_bisenet")
        on_cpu = not self.mm.use_gpu
        self.parser = FaceParser(
            path, self.mm.analyser_providers(), info, self.mm.session_options(on_cpu)
        )
        self.parser.load()
        return self.parser

    def _build_face_mask(self, frame, target_face, affine, size, face_crop):
        """Construye la máscara de fusión en espacio alineado según ``mask_mode``."""
        s = self.settings
        mode = s.mask_mode

        if mode == config.MASK_PARSING:
            try:
                parser = self._ensure_parser()
                aff512 = affine if size == 512 else imageutil.scale_affine(affine, 512.0 / size)
                target_aligned = self._aligned_crop(frame, aff512, 512)
                masks = parser.region_masks(target_aligned)
                fm = cv2.resize(masks["face"], (size, size), interpolation=cv2.INTER_LINEAR)
                fm = imageutil._feather(fm, s.mask_blur * 0.5)
                if fm.max() > 0.2:
                    return np.clip(fm, 0.0, 1.0)
            except Exception as exc:  # pragma: no cover - depende del modelo
                log.warning("Face parsing no disponible (%s); uso contorno de landmarks.", exc)

        if mode in (config.MASK_HULL, config.MASK_PARSING):
            lmk = getattr(target_face, "landmark_2d_106", None)
            if lmk is not None:
                pts = imageutil.transform_points(lmk, affine)
                return imageutil.convex_hull_mask(pts, size, blur=s.mask_blur, padding=s.mask_padding)
            kps_a = imageutil.transform_points(target_face.kps, affine)
            return imageutil.ellipse_face_mask(kps_a, size, blur=s.mask_blur, padding=s.mask_padding)

        if mode == config.MASK_ELLIPSE:
            kps_a = imageutil.transform_points(target_face.kps, affine)
            return imageutil.ellipse_face_mask(kps_a, size, blur=s.mask_blur, padding=s.mask_padding)

        return imageutil.build_soft_mask(size, size, blur=s.mask_blur, padding=s.mask_padding)

    def _swap_one(self, frame: np.ndarray, target_face) -> np.ndarray:
        s = self.settings
        bgr_fake, affine = self.swapper.swap_raw(frame, target_face, self.store.source_face)
        in_size = max(1, self.swapper.input_size)

        # Realce global con el enhancer (a 512) o swap crudo (a 128).
        if self.enhancer is not None:
            up = cv2.resize(bgr_fake, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            enhanced = self.enhancer.run(up, fidelity=s.codeformer_fidelity)
            blend = float(np.clip(s.enhancer_blend, 0.0, 1.0))
            face = cv2.addWeighted(enhanced, blend, up, 1.0 - blend, 0)
            affine = imageutil.scale_affine(affine, 512.0 / in_size)
            size = 512
        else:
            face = bgr_fake
            size = in_size

        # Realce DIRIGIDO de ojos y boca (regiones derivadas de los kps alineados).
        # Devuelve ojos vivos y dientes nítidos al cantar sin aplanar el resto.
        kps_aligned = imageutil.transform_points(target_face.kps, affine)
        eyes_m, mouth_m = imageutil.eye_mouth_region_masks(
            kps_aligned, size, mouth_open_boost=self._mouth_open_boost()
        )
        if s.eye_preservation > 0:
            face = imageutil.apply_local_detail(face, eyes_m, amount=s.eye_preservation * 1.2)
        if s.mouth_detail > 0:
            face = imageutil.apply_local_detail(face, mouth_m, amount=s.mouth_detail * 1.2)

        if s.color_match:
            ref = self._aligned_crop(frame, affine, size)
            face = imageutil.apply_color_transfer(face, ref)

        # Máscara que sigue el contorno real → sin deformar mandíbula/oreja en perfiles.
        mask = self._build_face_mask(frame, target_face, affine, size, face)
        return imageutil.paste_back_with_mask(frame, face, affine, mask, opacity=s.face_opacity)

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
        try:
            if self.settings.two_pass_temporal:
                processed = self._run_two_pass(video_path, info, writer, progress)
            else:
                processed = self._run_single_pass(video_path, info, writer, progress)
        finally:
            writer.close()

        return self._finalize(tmp_video, video_path, info, final_out, processed, progress)

    def _finalize(self, tmp_video, video_path, info, final_out, processed, progress) -> str:
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

    def _emit_progress(self, progress, processed, total, start, prefix=""):
        if not progress:
            return
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = (total - processed) / rate if rate > 0 else 0.0
        frac = min(processed / total, 0.99)
        progress(frac, f"{prefix}Frame {processed}/{total} · {rate:.1f} fps · ETA {format_eta(remaining)}")

    # --- 1 pasada: decodificar || GPU || codificar (buffers en RAM) -----------
    def _run_single_pass(self, video_path, info, writer, progress) -> int:
        prefetch, wq = self.mm.buffer_sizes((info.height, info.width))
        log.info("1 pasada · buffers RAM: prefetch=%d, escritura=%d", prefetch, wq)
        in_q: "queue.Queue" = queue.Queue(maxsize=prefetch)
        out_q: "queue.Queue" = queue.Queue(maxsize=wq)
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
                out_q.put(self._process_frame(frame, use_smoothing=True))
                processed += 1
                self._emit_progress(progress, processed, total, start)
        finally:
            out_q.put(None)
            wt.join()
            dt.join(timeout=2)

        if "decode" in errors:
            raise errors["decode"]
        if "write" in errors:
            raise errors["write"]
        return processed

    # --- 2 pasadas por tramos en RAM: analizar+suavizar, luego renderizar ------
    def _run_two_pass(self, video_path, info, writer, progress) -> int:
        chunk = self.mm.two_pass_chunk((info.height, info.width))
        total = max(1, info.frame_count)
        log.info("2 pasadas · tramo de %d frames en RAM", chunk)
        processed = 0
        start = time.time()

        def flush(frames, targets_per_frame):
            nonlocal processed
            # Suavizado centrado (no causal) de los landmarks del tramo.
            apply_two_pass_smoothing(
                targets_per_frame, time_sigma=2.0, motion_adaptive=self.settings.motion_adaptive
            )
            for fr, targets in zip(frames, targets_per_frame):
                for tf in targets:
                    fr = self._swap_one(fr, tf)
                writer.write(fr)
                processed += 1
                self._emit_progress(progress, processed, total, start, prefix="Render · ")

        frames: list = []
        targets_per_frame: list = []
        seen = 0
        for frame in videoutil.read_frames(video_path):
            work, _ = imageutil.limit_resolution(frame, self.settings.processing_resolution)
            faces = self.analyser.get_faces(work)
            targets = list(self.store.select_targets(faces))
            frames.append(work)
            targets_per_frame.append(targets)
            seen += 1
            if progress and seen % 15 == 0:
                progress(min(seen / total, 0.49) * 0.5, f"Analizando movimiento · {seen}/{total}")
            if len(frames) >= chunk:
                flush(frames, targets_per_frame)
                frames, targets_per_frame = [], []

        if frames:
            flush(frames, targets_per_frame)
        return processed
