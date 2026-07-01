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
        """Genera el vídeo (con audio si está activado). Devuelve dict con rutas."""
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

        # 1) Subir imagen
        p(0.01, "Subiendo imagen a ComfyUI…")
        try:
            image_ref = self.client.upload_image(image_path)
        except ComfyUIError as exc:
            raise I2VGenerationError(f"No se pudo subir la imagen: {exc}") from exc

        # 2) Generar vídeo
        p(0.03, "Preparando workflow de vídeo (Wan 2.2 I2V)…")
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
        video_out = self._run_graph(
            graph, media="video", phase=(0.05, 0.72),
            phase_msg="Generando vídeo (esto tarda varios minutos en 8 GB)…",
            progress=p,
        )
        ts = time.strftime("%Y%m%d_%H%M%S")
        video_tmp = TEMP_DIR / f"i2v_{ts}{Path(video_out.filename).suffix or '.mp4'}"
        self.client.download(video_out, str(video_tmp))

        # 3) Audio (opcional)
        audio_tmp: Optional[Path] = None
        if s.audio_enabled:
            try:
                audio_tmp = self._generate_audio(prompt, ts, progress=p)
            except Exception as exc:  # el audio no debe tumbar el resultado
                log.warning("Fallo al generar audio: %s", exc)
                p(0.90, "Audio no disponible; entrego el vídeo sin sonido.")

        # 4) Mezcla / salida final
        p(0.92, "Finalizando…")
        final = I2V_OUTPUT_DIR / f"i2v_{ts}.mp4"
        muxed = False
        if audio_tmp and audio_tmp.exists():
            muxed = videoutil.mux_external_audio(str(video_tmp), str(audio_tmp), str(final))
        if not muxed:
            # Sin audio: copiamos el vídeo tal cual a la carpeta de salida.
            final.write_bytes(Path(video_tmp).read_bytes())
        p(1.0, "¡Listo!")

        return {
            "video": str(final),
            "has_audio": muxed,
            "seconds": round(s.length_frames / max(1, s.fps), 1),
            "resolution": f"{s.width}x{s.height}",
        }

    def _generate_audio(self, video_prompt: str, ts: str, *, progress: ProgressCb) -> Optional[Path]:
        s = self.settings
        if progress:
            progress(0.74, "Generando audio (Stable Audio Open)…")
        graph = wf.load_workflow(WF_STABLE_AUDIO)
        audio_prompt = (s.audio_prompt or "").strip() or video_prompt
        graph = wf.patch_audio(
            graph, prompt=audio_prompt, negative=s.audio_negative,
            seconds=s.audio_seconds, seed=s.audio_seed,
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

        outputs = self.client.wait(prompt_id, progress=sub, timeout=GENERATION_TIMEOUT_S)
        result = self.client.pick_output(outputs, media)
        if result is None:
            raise I2VGenerationError(
                f"ComfyUI terminó pero no devolvió ningún fichero de {media}. "
                "Revisa la consola de ComfyUI (¿faltan modelos o nodos?)."
            )
        return result
