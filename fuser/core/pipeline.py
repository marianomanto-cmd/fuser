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
        log.info("Capacidades del motor: %s", self.engine.get_capabilities())

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
        """El **pipeline** decide 1 vs 2 pasadas consultando al motor + la RAM.

        - Si el motor no las soporta → 1 pasada.
        - Si el usuario las activó → 2 pasadas.
        - FaceFusion + Modo Videos Musicales → se **fuerzan** 2 pasadas si hay RAM
          suficiente (prioriza calidad), aunque el usuario no las marcara.
        """
        if not self.engine.supports_two_pass():
            return False
        if self.settings.two_pass_temporal:
            return True
        music = self.settings.expression_mode in (config.EXPR_MUSIC_VIDEO, config.EXPR_HIGH_EXPRESSION)
        if music and self.engine.prefers_two_pass():
            return bool(self.mm.get_recommended_mode()["two_pass"])
        return False

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

        # El pipeline decide el flujo consultando las capacidades del motor.
        caps = self.engine.get_capabilities()
        music = self.settings.expression_mode in (config.EXPR_MUSIC_VIDEO, config.EXPR_HIGH_EXPRESSION)
        region_enh = bool(caps.get("region_enhancement") and music and self.settings.mouth_enhancer)
        log.info(
            "Flujo: motor=%s · 2 pasadas=%s · realce localizado de boca=%s",
            self.settings.engine, self._use_two_pass(), region_enh,
        )
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

        # Segunda pasada opcional: detecta frames defectuosos y los corrige.
        if self.settings.qc_second_pass:
            tmp_video = Path(self._run_qc_pass(video_path, tmp_video, info, progress))

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

    # ----- 2ª pasada: detección + corrección de defectos ----------------------
    def _run_qc_pass(self, video_path, swapped_video, info, progress: ProgressCb):
        """Detecta frames defectuosos del vídeo swapeado y los corrige con el MISMO
        modelo (re-swap con detección agresiva o relleno temporal). Devuelve la ruta
        del vídeo (corregido, o el original si no hubo nada que arreglar)."""
        from . import qc_pass
        from ..models.face_analyser import FaceAnalyser

        proc_res = self.settings.processing_resolution
        orig = [imageutil.limit_resolution(f, proc_res)[0] for f in videoutil.read_frames(video_path)]
        swap = list(videoutil.read_frames(str(swapped_video)))
        n = min(len(orig), len(swap))
        orig, swap = orig[:n], swap[:n]
        if n < 8:
            return str(swapped_video)

        # Guardia de RAM: el QC mantiene ~2 juegos de frames en memoria.
        h, w = swap[0].shape[:2]
        need_gb = 2.2 * n * h * w * 3 / (1024 ** 3)
        avail = self.mm.info.ram_available_gb or 8.0
        if need_gb > 0.45 * avail:
            log.warning("QC: vídeo demasiado largo para la RAM (~%.1f GB > presupuesto). Se "
                        "omite la 2ª pasada; cortá el vídeo en partes para usarla.", need_gb)
            if progress:
                progress(0.99, "2ª pasada omitida (vídeo largo): cortalo en partes.")
            return str(swapped_video)

        if progress:
            progress(0.0, "🔍 Segunda pasada: analizando defectos…")
        analyser = FaceAnalyser(self.mm.analyser_providers(), self.mm.ctx_id(), self.mm.det_size)
        analyser.load()
        metrics = qc_pass.analyze(
            orig, swap, analyser,
            progress=lambda f, m="": progress and progress(0.45 * f, m),
        )
        defects = qc_pass.flag_defects(metrics, self.settings.qc_sensitivity)
        if not defects:
            log.info("QC: sin defectos en %d frames.", n)
            if progress:
                progress(1.0, "✅ Segunda pasada: no se encontraron defectos.")
            return str(swapped_video)

        reswap_fn = self._recovery_reswap_fn()
        report = qc_pass.correct(
            orig, swap, metrics, defects, analyser, reswap_fn=reswap_fn,
            progress=lambda f, m="": progress and progress(0.45 + 0.45 * f, m),
        )
        self._restore_after_recovery()
        log.info("QC: %s", report.summary())

        out2 = TEMP_DIR / f"fuser_qc_{int(time.time())}.mp4"
        writer = videoutil.FFmpegVideoWriter(
            str(out2), swap[0].shape[1], swap[0].shape[0], info.fps,
            crf=self.settings.output_quality, encoder=self.settings.output_video_encoder,
        )
        try:
            for fr in swap:
                writer.write(fr)
        finally:
            writer.close()
        if progress:
            progress(1.0, f"✅ 2ª pasada: {report.summary()}")
        return str(out2)

    def _recovery_reswap_fn(self):
        """Reconfigura el motor con DETECCIÓN AGRESIVA (mismo modelo) y devuelve
        ``fn(frame)->swap``. Restaurar con ``_restore_after_recovery``."""
        from dataclasses import replace

        self._qc_saved_settings = self.settings
        recovery = replace(
            self.settings, ff_detector_angles=(0, 90, 180, 270),
            ff_detector_score=0.25, ff_landmarker_score=0.15, ff_temporal_fallback=True,
        )
        try:
            self.engine.update_runtime(recovery)
        except Exception as exc:  # pragma: no cover
            log.warning("QC: no pude configurar el re-swap de recuperación: %s", exc)
            return None

        def _fn(frame):
            return self.engine.process_frame(frame, use_smoothing=False)

        return _fn

    def _restore_after_recovery(self) -> None:
        if getattr(self, "_qc_saved_settings", None) is not None:
            try:
                self.engine.update_runtime(self._qc_saved_settings)
            except Exception:  # pragma: no cover
                pass
            self._qc_saved_settings = None

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
        start = time.time()
        proc_res = self.settings.processing_resolution

        # Progreso MONÓTONO con ETA real combinando las dos fases (análisis +
        # render). El render es bastante más caro que el análisis, así que pesa
        # más: frac = (detectados*0.2 + renderizados*0.8) / total. Así la barra
        # nunca retrocede y la ETA se autocorrige entre fases.
        detected = 0
        rendered = 0
        det_w, ren_w = 0.2, 0.8

        def emit(phase: str = "render") -> None:
            if not progress:
                return
            frac = min((detected * det_w + rendered * ren_w) / total, 0.99)
            elapsed = time.time() - start
            remaining = elapsed * (1.0 - frac) / frac if frac > 0.01 else 0.0
            if phase == "detect":
                progress(frac, f"Analizando movimiento · {detected}/{total} · "
                               f"ETA {format_eta(remaining)}")
            else:
                fps = rendered / elapsed if elapsed > 0 else 0.0
                progress(frac, f"Procesando · frame {rendered}/{total} · {fps:.1f} fps · "
                               f"ETA {format_eta(remaining)}")

        def flush(frames, faces_per_frame):
            nonlocal rendered
            apply_two_pass_smoothing(
                faces_per_frame, time_sigma=time_sigma, motion_adaptive=self.settings.motion_adaptive
            )
            for fr, faces in zip(frames, faces_per_frame):
                targets = self.engine.select_targets(faces)
                writer.write(self.engine.render(fr, targets))
                rendered += 1
                emit()

        frames: list = []
        faces_per_frame: list = []
        for frame in videoutil.read_frames(video_path):
            work, _ = imageutil.limit_resolution(frame, proc_res)
            faces = self.engine.detect(work)
            frames.append(work)
            faces_per_frame.append(faces)
            detected += 1
            if detected % 10 == 0:
                emit("detect")
            if len(frames) >= chunk:
                flush(frames, faces_per_frame)
                frames, faces_per_frame = [], []

        if frames:
            flush(frames, faces_per_frame)
        return rendered
