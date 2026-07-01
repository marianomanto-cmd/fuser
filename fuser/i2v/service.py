"""Orquestación de *Imagen → Vídeo* (agnóstica a la UI).

``I2VService`` une todas las piezas:

    imagen + prompt
        -> sube imagen a ComfyUI
        -> genera vídeo (Wan 2.2 I2V GGUF, sin audio)
        -> genera audio (Stable Audio Open)        [opcional]
        -> mezcla vídeo + audio con ffmpeg
        -> devuelve un .mp4 final en outputs/i2v/

No importa nada pesado: solo ``fuser`` + librería estándar. La UI llama a
``generate`` con un callback de progreso.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..utils import video as videoutil
from ..utils.logging import get_logger
from . import models as i2v_models
from . import workflow as wf
from .comfy_client import ComfyUIClient, ComfyUIError, ComfyUINotAvailable, OutputFile
from .config import (
    GENERATION_TIMEOUT_S,
    I2V_OUTPUT_DIR,
    TEMP_DIR,
    WF_STABLE_AUDIO,
    I2VSettings,
    ensure_i2v_dirs,
)

log = get_logger(__name__)

ProgressCb = Optional[Callable[[float, str], None]]


class I2VGenerationError(RuntimeError):
    """Error de alto nivel durante la generación (mensaje accionable)."""


class I2VService:
    """Servicio de generación Imagen→Vídeo contra un ComfyUI local."""

    def __init__(self, settings: I2VSettings):
        self.settings = settings.resolved()
        self._client: Optional[ComfyUIClient] = None

    @property
    def client(self) -> ComfyUIClient:
        if self._client is None:
            self._client = ComfyUIClient(self.settings.comfy_url, timeout=20)
        return self._client

    # ------------------------------------------------------------------ checks
    def ensure_available(self) -> None:
        if not self.client.is_available():
            raise ComfyUINotAvailable(self.settings.comfy_url)

    def _required_node_types(self, include_audio: bool) -> List[str]:
        types: set = set()
        for name in (self.settings.workflow, WF_STABLE_AUDIO if include_audio else None):
            if not name:
                continue
            try:
                g = wf.load_workflow(name)
            except Exception:
                continue
            for node in g.values():
                ct = node.get("class_type")
                if ct:
                    types.add(ct)
        return sorted(types)

    def validate_setup(self) -> Dict:
        """Diagnóstico: ¿ComfyUI vivo? ¿nodos? ¿modelos? Devuelve dict + markdown."""
        reachable = self.client.is_available()
        object_info = None
        if reachable:
            try:
                object_info = self.client.get_object_info()
            except Exception as exc:
                log.warning("No pude leer /object_info: %s", exc)

        node_types = self._required_node_types(include_audio=self.settings.audio_enabled)
        nodes = {ct: (object_info is not None and ct in object_info) for ct in node_types}
        model_rows = i2v_models.model_report(object_info, include_audio=self.settings.audio_enabled)

        report = {
            "reachable": reachable,
            "url": self.settings.comfy_url,
            "nodes": nodes,
            "models": model_rows,
            "object_info": object_info is not None,
        }
        report["markdown"] = self._format_report_md(report)
        return report

    @staticmethod
    def _format_report_md(report: Dict) -> str:
        lines = ["### 🔌 Estado de ComfyUI (Imagen → Vídeo)", ""]
        if report["reachable"]:
            lines.append(f"- **Servidor:** ✅ conectado en `{report['url']}`")
        else:
            lines.append(f"- **Servidor:** ❌ sin respuesta en `{report['url']}`")
            lines.append("  - Arráncalo: `python main.py --listen 127.0.0.1 --port 8188 --lowvram`")
            return "\n".join(lines)

        if not report["object_info"]:
            lines.append("- ⚠️ No pude leer `/object_info` (¿versión rara de ComfyUI?).")
            return "\n".join(lines)

        missing_nodes = [n for n, ok in report["nodes"].items() if not ok]
        if missing_nodes:
            lines.append("- **Custom nodes que faltan:** ❌ " + ", ".join(f"`{n}`" for n in missing_nodes))
            lines.append("  - Instálalos desde *ComfyUI-Manager* (ComfyUI-GGUF, VideoHelperSuite, MultiGPU).")
        else:
            lines.append("- **Custom nodes:** ✅ todos presentes")

        present, unknown, absent = [], [], []
        for row in report["models"]:
            m, ok = row["model"], row["present"]
            (present if ok else (absent if ok is False else unknown)).append(m.filename)
        if absent:
            lines.append("- **Modelos que faltan:** ❌ " + ", ".join(f"`{f}`" for f in absent))
            lines.append("  - Descárgalos con `python scripts/setup_i2v.py --download` "
                         "o colócalos a mano (ver `docs/IMAGE_TO_VIDEO.md`).")
        if present:
            lines.append(f"- **Modelos presentes:** ✅ {len(present)}")
        if unknown:
            lines.append("- **Modelos:** ⚠️ no pude verificar " + ", ".join(f"`{f}`" for f in unknown))

        ready = report["reachable"] and not missing_nodes and not absent
        lines.append("")
        lines.append("✅ **Listo para generar.**" if ready else
                     "⚠️ **Aún faltan piezas** (mira arriba) antes de generar.")
        return "\n".join(lines)

    # --------------------------------------------------------------- generate
    def generate(self, image_path: str, prompt: str, negative: str = "",
                 *, progress: ProgressCb = None) -> Dict:
        """Genera el vídeo. Si ``settings.n_clips`` > 1 ENCADENA clips (el último
        frame de cada clip es la imagen de arranque del siguiente) y los une en un
        único vídeo. En 8 GB encadenar clips cortos es la vía FIABLE para ~10 s
        (una sola pasada larga suele petar el VAE). Devuelve dict con rutas."""
        if not image_path:
            raise I2VGenerationError("Sube una imagen de entrada.")
        if not (prompt or "").strip():
            raise I2VGenerationError("Escribe un prompt que describa el movimiento/escena.")
        ensure_i2v_dirs()
        s = self.settings

        # Resolución AUTO por aspecto de la imagen (width/height <= 0). Evita que Wan
        # haga center-crop de una foto vertical/cuadrada en un marco apaisado y
        # "invente" el resto de la escena (el bug de "cambia toda la imagen").
        if s.width <= 0 or s.height <= 0:
            from .config import bucket_for_image
            s.width, s.height = bucket_for_image(image_path)
            log.info("i2v auto-res: %dx%d (según el aspecto de tu imagen)", s.width, s.height)

        def p(frac: float, msg: str = "") -> None:
            if progress:
                progress(max(0.0, min(1.0, frac)), msg)

        self.ensure_available()
        n = self._n_clips()
        ts = time.strftime("%Y%m%d_%H%M%S")
        tmps: List[Path] = []
        try:
            clips = self._render_chain(image_path, prompt, negative, n=n, ts=ts,
                                       phase=(0.02, 0.72), progress=p, tmps=tmps)
            video_tmp = self._join_clips(clips, ts, progress=p, tmps=tmps)
            return self._finalize(video_tmp, prompt, ts, progress=p, tmps=tmps,
                                  resolution=f"{s.width}x{s.height}")
        finally:
            self._cleanup(tmps)

    # ---------------------------------------------------------------- extend
    def extend(self, base_video: str, prompt: str, negative: str = "",
               *, progress: ProgressCb = None) -> Dict:
        """Extiende un vídeo YA generado: toma su ÚLTIMO frame como imagen de
        arranque de ``settings.n_clips`` clips nuevos y los pega detrás del vídeo
        base (uno tras otro). La continuación usa la MISMA resolución que el base."""
        if not base_video or not Path(base_video).exists():
            raise I2VGenerationError("No hay un vídeo base para extender (genera uno primero).")
        if not (prompt or "").strip():
            raise I2VGenerationError("Escribe un prompt para la continuación (describe el movimiento).")
        ensure_i2v_dirs()
        s = self.settings
        info = videoutil.probe(base_video)
        s.width, s.height = info.width, info.height   # unir sin reescalar

        def p(frac: float, msg: str = "") -> None:
            if progress:
                progress(max(0.0, min(1.0, frac)), msg)

        self.ensure_available()
        n = self._n_clips()
        ts = time.strftime("%Y%m%d_%H%M%S")
        tmps: List[Path] = []
        try:
            p(0.02, "Tomando el último frame del vídeo…")
            seed = TEMP_DIR / f"i2v_{ts}_seedbase.png"
            videoutil.save_last_frame(base_video, str(seed)); tmps.append(seed)
            clips = self._render_chain(str(seed), prompt, negative, n=n, ts=ts,
                                       phase=(0.05, 0.72), progress=p, tmps=tmps)
            # Une el vídeo base + los clips nuevos (force: siempre reencoda para pegar).
            video_tmp = self._join_clips([base_video, *clips], ts, progress=p,
                                         tmps=tmps, force=True)
            return self._finalize(video_tmp, prompt, ts, progress=p, tmps=tmps,
                                  resolution=f"{info.width}x{info.height}")
        finally:
            self._cleanup(tmps)

    # ------------------------------------------------------------- internos
    def _n_clips(self) -> int:
        from .config import MAX_N_CLIPS
        return max(1, min(MAX_N_CLIPS, int(getattr(self.settings, "n_clips", 1) or 1)))

    def _render_chain(self, start_image: str, prompt: str, negative: str, *,
                      n: int, ts: str, phase: tuple, progress: ProgressCb,
                      tmps: List[Path]) -> List[Path]:
        """Renderiza ``n`` clips encadenados: último frame -> arranque del siguiente."""
        lo, hi = phase
        clips: List[Path] = []
        cur = start_image
        for i in range(n):
            a = lo + (hi - lo) * (i / n)
            b = lo + (hi - lo) * ((i + 1) / n)
            msg = (f"Generando clip {i+1}/{n}…" if n > 1
                   else "Generando vídeo (varios minutos en 8 GB)…")
            clip = self._render_clip(cur, prompt, negative, tag=f"{ts}_{i}",
                                     phase=(a, b), phase_msg=msg, progress=progress)
            clips.append(clip); tmps.append(clip)
            if i < n - 1:
                # Frame de enlace SIN pérdida (PNG); solo se reencoda al unir al final.
                seed = TEMP_DIR / f"i2v_{ts}_seed{i}.png"
                videoutil.save_last_frame(str(clip), str(seed))
                cur = str(seed); tmps.append(seed)
        return clips

    def _render_clip(self, image_path: str, prompt: str, negative: str, *, tag: str,
                     phase: tuple, phase_msg: str, progress: ProgressCb) -> Path:
        """Sube la imagen, parchea el workflow y genera UN clip. Devuelve la ruta tmp."""
        s = self.settings
        try:
            image_ref = self.client.upload_image(str(image_path))
        except ComfyUIError as exc:
            raise I2VGenerationError(f"No se pudo subir la imagen: {exc}") from exc
        try:
            graph = wf.load_workflow(s.workflow)
        except Exception as exc:
            raise I2VGenerationError(f"No se pudo cargar el workflow '{s.workflow}': {exc}") from exc
        graph = wf.patch_i2v(
            graph, image=image_ref, positive=prompt, negative=negative or "",
            width=s.width, height=s.height, length=s.length_frames, fps=s.fps,
            steps=s.steps, cfg=s.cfg, seed=s.seed, sampler=s.sampler,
            scheduler=s.scheduler, shift=s.shift,
            high_model=s.high_noise_model, low_model=s.low_noise_model,
            virtual_vram_gb=s.virtual_vram_gb,
        )
        out = self._run_graph(graph, media="video", phase=phase,
                              phase_msg=phase_msg, progress=progress)
        dest = TEMP_DIR / f"i2v_{tag}{Path(out.filename).suffix or '.mp4'}"
        self.client.download(out, str(dest))
        return dest

    def _join_clips(self, clips: List, ts: str, *, progress: ProgressCb,
                    tmps: List[Path], force: bool = False) -> Path:
        """Une clips (con ``drop_seam``). 1 clip y sin ``force`` -> se devuelve tal cual."""
        paths = [Path(c) for c in clips]
        if len(paths) == 1 and not force:
            return paths[0]
        progress(0.73, "Uniendo los clips…")
        joined = TEMP_DIR / f"i2v_{ts}_joined.mp4"
        if not videoutil.concat_videos([str(c) for c in paths], str(joined), drop_seam=True):
            raise I2VGenerationError("No se pudieron unir los clips (¿ffmpeg en el PATH?).")
        tmps.append(joined)
        return joined

    def _finalize(self, video_tmp: Path, prompt: str, ts: str, *,
                  progress: ProgressCb, tmps: List[Path], resolution: str) -> Dict:
        """Audio (opcional, para la duración REAL del vídeo) + copia a outputs/."""
        info = videoutil.probe(str(video_tmp))
        audio_tmp: Optional[Path] = None
        note = ""
        if self.settings.audio_enabled:
            try:
                audio_tmp = self._generate_audio(prompt, ts, seconds=info.duration + 0.2,
                                                 progress=progress)
                if audio_tmp:
                    tmps.append(audio_tmp)
            except Exception as exc:  # el audio no debe tumbar el resultado
                log.warning("Fallo al generar audio: %s", exc)
                note = "Audio no disponible (falló su generación); vídeo sin sonido."
                progress(0.90, "Audio no disponible; entrego el vídeo sin sonido.")

        progress(0.92, "Finalizando…")
        final = I2V_OUTPUT_DIR / f"i2v_{ts}.mp4"
        muxed = False
        if audio_tmp and audio_tmp.exists():
            muxed = videoutil.mux_external_audio(str(video_tmp), str(audio_tmp), str(final))
            if not muxed:
                log.warning("El audio se generó pero el mux con ffmpeg falló; vídeo sin "
                            "sonido. ¿ffmpeg instalado y en PATH?")
                note = "Audio generado pero no se pudo mezclar (¿ffmpeg en PATH?)."
        if not muxed:
            shutil.copyfile(str(video_tmp), str(final))
        progress(1.0, "¡Listo!")
        return {
            "video": str(final),
            "has_audio": muxed,
            "seconds": round(info.duration, 1),
            "resolution": resolution,
            "note": note,
        }

    def _cleanup(self, tmps: List[Path]) -> None:
        for tmp in tmps:
            try:
                if tmp and Path(tmp).exists():
                    Path(tmp).unlink()
            except Exception:
                pass

    def _generate_audio(self, video_prompt: str, ts: str, *, seconds: float,
                        progress: ProgressCb) -> Optional[Path]:
        s = self.settings
        if progress:
            progress(0.74, "Generando audio (Stable Audio Open)…")
        graph = wf.load_workflow(WF_STABLE_AUDIO)
        audio_prompt = (s.audio_prompt or "").strip() or video_prompt
        graph = wf.patch_audio(
            graph, prompt=audio_prompt, negative=s.audio_negative,
            seconds=float(seconds), seed=s.audio_seed,
            steps=s.audio_steps, cfg=s.audio_cfg,
        )
        out = self._run_graph(
            graph, media="audio", phase=(0.75, 0.90),
            phase_msg="Generando audio…", progress=progress,
        )
        suffix = Path(out.filename).suffix or ".flac"
        dest = TEMP_DIR / f"i2v_{ts}_audio{suffix}"
        self.client.download(out, str(dest))
        return dest

    def _run_graph(self, graph: dict, *, media: str, phase: tuple,
                   phase_msg: str, progress: ProgressCb) -> OutputFile:
        lo, hi = phase
        if progress:
            progress(lo, phase_msg)
        try:
            prompt_id = self.client.queue_prompt(graph)
        except ComfyUIError as exc:
            raise I2VGenerationError(str(exc)) from exc

        def sub(frac: float, msg: str = "") -> None:
            if progress:
                progress(lo + (hi - lo) * frac, msg or phase_msg)

        try:
            outputs = self.client.wait(prompt_id, progress=sub, timeout=GENERATION_TIMEOUT_S)
        except ComfyUIError as exc:
            # Si abandonamos (timeout u otro error) NO dejes el job huérfano
            # ocupando la GPU: en 8 GB bloquearía el siguiente intento / el swap.
            try:
                self.client.interrupt()
            except Exception:
                pass
            raise I2VGenerationError(str(exc)) from exc
        result = self.client.pick_output(outputs, media)
        if result is None:
            raise I2VGenerationError(
                f"ComfyUI terminó pero no devolvió ningún fichero de {media}. "
                "Revisa la consola de ComfyUI (¿faltan modelos o nodos?)."
            )
        return result
