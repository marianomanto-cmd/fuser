"""Orquestación del face swap de vídeo — **agnóstica al motor**.

El pipeline no conoce a InsightFace ni a FaceFusion: habla con la interfaz
``BaseFaceSwapper`` (ver ``fuser/engines``). Se encarga de lo común a ambos
motores: I/O de vídeo, buffers en RAM, progreso/ETA, audio y los dos modos de
recorrido (1 pasada solapada, o 2 pasadas por tramos en RAM si el motor lo
soporta).

    [hilo decodificador] --in_queue--> [GPU: motor] --out_queue--> [hilo escritor]
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .. import config
from ..config import OUTPUTS_DIR, TEMP_DIR, Settings, ensure_dirs
from ..engines import create_engine
from ..utils import image as imageutil
from ..utils import video as videoutil
from ..utils.logging import get_logger
from .memory_manager import MemoryManager
from .temporal import apply_two_pass_smoothing

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
    """Pipeline de face swap, configurado por ``Settings``; delega en un motor."""

    def __init__(self, settings: Settings):
        self.settings = settings.resolved()
        self.mm = MemoryManager(self.settings)
        self.engine = None
        self._loaded = False

    # ----- carga de modelos ----------------------------------------------------
    def load_models(self, progress: ProgressCb = None) -> None:
        if self._loaded:
            return
        ensure_dirs()
        self.engine = create_engine(self.settings, self.mm)
        self.engine.load(progress=progress)
        self._loaded = True
        log.info("Pipeline listo · motor=%s · %s", self.settings.engine, self.mm.summary())

    @property
    def loaded(self) -> bool:
        return self._loaded

    def model_signature(self) -> tuple:
        """Identidad de los ajustes que obligan a recargar modelos/motor."""
        s = self.settings
        return (
            s.engine, s.ff_swapper_model, s.ff_pixel_boost,
            s.swapper_model, s.enhancer_model, s.memory_mode,
            s.force_cpu, round(s.gpu_mem_limit_gb, 2), s.det_size,
        )

    def update_runtime(self, settings: Settings) -> None:
        self.settings = settings.resolved()
        if self.engine is not None:
            self.engine.update_runtime(self.settings)

    # ----- fuente / referencia -------------------------------------------------
    def prepare_source(self, images: List[np.ndarray]):
        if not self._loaded:
            self.load_models()
        return self.engine.prepare_source(images)

    def set_reference_from_frame(self, frame: np.ndarray, face_index: int = 0) -> bool:
        if not self._loaded:
            self.load_models()
        return self.engine.set_reference(frame, face_index)

    # ----- utilidades ----------------------------------------------------------
    def _output_dims(self, w: int, h: int) -> Tuple[int, int]:
        mx = self.settings.processing_resolution
        if mx <= 0 or max(w, h) <= mx:
            return w, h
        f = mx / float(max(w, h))
        return int(round(w * f)), int(round(h * f))

    def _use_two_pass(self) -> bool:
        return bool(self.settings.two_pass_temporal and self.engine.supports_two_pass())

    # ----- previsualización ----------------------------------------------------
    def preview(
        self, video_path: str, n_frames: int = 6, progress: ProgressCb = None
    ) -> List[Tuple[np.ndarray, str]]:
        if not self._loaded:
            self.load_models()
        if not self.engine.has_source:
            raise RuntimeError("Sube primero una imagen fuente con una cara visible.")

        info = videoutil.probe(video_path)
        indices = videoutil.keyframe_indices(info.frame_count, n_frames)
        frames = videoutil.get_frames_at(video_path, indices)

        results: List[Tuple[np.ndarray, str]] = []
        for i, (idx, frame) in enumerate(zip(indices, frames)):
            if progress:
                progress((i + 1) / max(1, len(frames)), f"Previsualizando frame {idx}…")
            work, _ = imageutil.limit_resolution(frame, self.settings.processing_resolution)
            out = self.engine.process_frame(work, use_smoothing=False)
            results.append((imageutil.to_rgb(out), f"Frame {idx}"))
        return results

    # ----- procesamiento completo ---------------------------------------------
    def process_video(
        self, video_path: str, output_path: Optional[str] = None, progress: ProgressCb = None
    ) -> str:
        if not self._loaded:
            self.load_models()
        if not self.engine.has_source:
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
            if self._use_two_pass():
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

    # --- 1 pasada: decodificar || motor (GPU) || codificar --------------------
    def _run_single_pass(self, video_path, info, writer, progress) -> int:
        prefetch, wq = self.mm.buffer_sizes((info.height, info.width))
        log.info("1 pasada · buffers RAM: prefetch=%d, escritura=%d", prefetch, wq)
        in_q: "queue.Queue" = queue.Queue(maxsize=prefetch)
        out_q: "queue.Queue" = queue.Queue(maxsize=wq)
        errors: dict = {}
        proc_res = self.settings.processing_resolution

        def decoder():
            try:
                for frame in videoutil.read_frames(video_path):
                    work, _ = imageutil.limit_resolution(frame, proc_res)
                    in_q.put(work)
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
        self.engine.reset_temporal()
        try:
            while True:
                frame = in_q.get()
                if frame is None:
                    break
                if "decode" in errors:
                    raise errors["decode"]
                out_q.put(self.engine.process_frame(frame, use_smoothing=True))
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

    # --- 2 pasadas por tramos en RAM (solo motores que lo soportan) -----------
    def _run_two_pass(self, video_path, info, writer, progress) -> int:
        chunk = self.mm.two_pass_chunk((info.height, info.width))
        total = max(1, info.frame_count)
        # Modo musical/alta expresión: ventana de estabilización más amplia
        # (el suavizado adaptativo mantiene la boca rápida sin "lag").
        music = self.settings.expression_mode in (config.EXPR_MUSIC_VIDEO, config.EXPR_HIGH_EXPRESSION)
        time_sigma = 3.0 if music else 2.0
        if self.mm.is_facefusion:
            time_sigma += 0.5  # mejor calidad base -> tolera suavizado más fuerte
        log.info("2 pasadas · tramo de %d frames en RAM · time_sigma=%.1f", chunk, time_sigma)
        processed = 0
        start = time.time()
        proc_res = self.settings.processing_resolution

        def flush(frames, faces_per_frame):
            nonlocal processed
            apply_two_pass_smoothing(
                faces_per_frame, time_sigma=time_sigma, motion_adaptive=self.settings.motion_adaptive
            )
            for fr, faces in zip(frames, faces_per_frame):
                targets = self.engine.select_targets(faces)
                writer.write(self.engine.render(fr, targets))
                processed += 1
                self._emit_progress(progress, processed, total, start, prefix="Render · ")

        frames: list = []
        faces_per_frame: list = []
        seen = 0
        for frame in videoutil.read_frames(video_path):
            work, _ = imageutil.limit_resolution(frame, proc_res)
            faces = self.engine.detect(work)
            frames.append(work)
            faces_per_frame.append(faces)
            seen += 1
            if progress and seen % 15 == 0:
                progress(min(seen / total, 0.49) * 0.5, f"Analizando movimiento · {seen}/{total}")
            if len(frames) >= chunk:
                flush(frames, faces_per_frame)
                frames, faces_per_frame = [], []

        if frames:
            flush(frames, faces_per_frame)
        return processed
