"""Interfaz Gradio de Fuser.

Optimizada para el caso de uso de **caras de mujeres en videos musicales**:
múltiples ángulos, boca muy abierta al cantar, expresiones intensas y mucho
movimiento. El **Modo** (arriba) ajusta de golpe enhancer, máscara, preservación
de ojos/boca, estabilidad temporal y nº de referencias recomendado.

El pipeline se cachea entre ejecuciones: solo se recargan modelos si cambian los
ajustes que afectan a modelos/memoria; el resto se aplica en caliente.

Diseño visual: tema "pro creative tool" (oscuro + acento cyan, fuente Inter,
layout de 3 columnas) según el handoff de diseño. Se aplica con un tema de
Gradio + CSS custom (ver ``CUSTOM_CSS``), sin tocar la lógica ni el wiring.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import gradio as gr
import numpy as np

from .. import __app_name__, __version__, config
from ..core.pipeline import SwapPipeline
from ..utils import video as videoutil
from ..utils.logging import get_logger
from ..utils.system import format_system_summary

try:
    # Pestaña Imagen → Vídeo (Wan 2.2 vía ComfyUI). Opcional: si algo falla en su
    # import, la app sigue funcionando sin esa pestaña.
    from .i2v_interface import build_i2v_tab
except Exception:  # pragma: no cover
    build_i2v_tab = None

log = get_logger(__name__)

_PIPELINE_CACHE: dict = {"pipeline": None, "signature": None}


def free_swap_vram() -> None:
    """Descarga el pipeline de swap cacheado y libera su VRAM.

    Clave para Imagen→Vídeo en 8 GB: el swap y ComfyUI comparten los 8 GB. Si el
    swap tiene modelos cargados en la GPU cuando ComfyUI genera vídeo, la VRAM se
    satura y la generación va ~50× más lenta (thrashing). La pestaña i2v llama a
    esto antes de generar, así el usuario no tiene que cerrar nada a mano.
    """
    pipe = _PIPELINE_CACHE.get("pipeline")
    if pipe is not None:
        try:
            if getattr(pipe, "engine", None) is not None:
                pipe.engine.unload()
        except Exception as exc:  # pragma: no cover
            log.warning("No pude descargar el pipeline de swap: %s", exc)
    _PIPELINE_CACHE["pipeline"] = None
    _PIPELINE_CACHE["signature"] = None
    try:
        import gc

        gc.collect()
    except Exception:
        pass

# Intentos por video en la cola: si falla, se manda al FINAL y se reintenta luego;
# tras este nº de intentos se descarta (evita un bucle infinito con un video imposible).
QUEUE_MAX_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_image(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _build_settings(
    engine, ff_swapper_model, ff_pixel_boost,
    use_mouth_pixel_boost, mouth_enhancement_strength, profile_blending_strength,
    swapper_model, enhancer_model, enhancer_blend, codeformer_fidelity,
    expression_mode,
    face_selector, reference_index, reference_distance, reference_count,
    face_opacity, mask_mode, mask_blur, mask_padding, eye_preservation, mouth_detail,
    mouth_enhancer, color_match, processing_resolution,
    temporal_smoothing, temporal_alpha, motion_adaptive, two_pass_temporal,
    memory_mode, gpu_mem_limit, force_cpu, ram_mode,
    keep_audio, keep_fps, output_quality,
    skin_detail,
    qc_second_pass, qc_sensitivity,
) -> config.Settings:
    return config.Settings(
        engine=engine,
        ff_swapper_model=ff_swapper_model,
        ff_pixel_boost=ff_pixel_boost,
        use_mouth_pixel_boost=bool(use_mouth_pixel_boost),
        mouth_enhancement_strength=float(mouth_enhancement_strength),
        profile_blending_strength=float(profile_blending_strength),
        swapper_model=swapper_model,
        enhancer_model=enhancer_model,
        enhancer_blend=float(enhancer_blend),
        codeformer_fidelity=float(codeformer_fidelity),
        expression_mode=expression_mode,
        face_selector=face_selector,
        reference_face_index=int(reference_index),
        reference_distance=float(reference_distance),
        reference_count=int(reference_count),
        face_opacity=float(face_opacity),
        mask_mode=mask_mode,
        mask_blur=float(mask_blur),
        mask_padding=float(mask_padding),
        eye_preservation=float(eye_preservation),
        mouth_detail=float(mouth_detail),
        mouth_enhancer=bool(mouth_enhancer),
        color_match=bool(color_match),
        processing_resolution=int(processing_resolution),
        temporal_smoothing=bool(temporal_smoothing),
        temporal_alpha=float(temporal_alpha),
        motion_adaptive=bool(motion_adaptive),
        two_pass_temporal=bool(two_pass_temporal),
        memory_mode=memory_mode,
        gpu_mem_limit_gb=float(gpu_mem_limit),
        force_cpu=bool(force_cpu),
        ram_mode=ram_mode,
        keep_audio=bool(keep_audio),
        keep_fps=bool(keep_fps),
        output_quality=int(output_quality),
        skin_detail=float(skin_detail),
        qc_second_pass=bool(qc_second_pass),
        qc_sensitivity=float(qc_sensitivity),
    )


def _get_pipeline(settings: config.Settings, progress=None) -> SwapPipeline:
    pipeline = SwapPipeline(settings)
    signature = pipeline.model_signature()
    cached = _PIPELINE_CACHE["pipeline"]
    if cached is not None and _PIPELINE_CACHE["signature"] == signature:
        cached.update_runtime(settings)
        return cached
    pipeline.load_models(progress=progress)
    _PIPELINE_CACHE["pipeline"] = pipeline
    _PIPELINE_CACHE["signature"] = signature
    return pipeline


def _prepare(pipeline: SwapPipeline, source_files, video_path) -> str:
    """Valida entradas, prepara la cara fuente (multi-ref) y devuelve un resumen."""
    if not source_files:
        raise gr.Error("Sube al menos una imagen fuente (la cara a aplicar).")
    if not video_path:
        raise gr.Error("Sube un vídeo objetivo.")

    images = []
    for f in source_files:
        path = f if isinstance(f, str) else getattr(f, "name", None)
        img = _read_image(path) if path else None
        if img is not None:
            images.append(img)
    if not images:
        raise gr.Error("No se pudieron leer las imágenes fuente.")

    try:
        stats = pipeline.prepare_source(images)
    except ValueError as exc:
        raise gr.Error(str(exc))

    if pipeline.settings.face_selector == config.FACE_SELECTOR_REFERENCE:
        info = videoutil.probe(video_path)
        mid = videoutil.get_frames_at(video_path, [info.frame_count // 2])
        if mid:
            pipeline.set_reference_from_frame(mid[0], pipeline.settings.reference_face_index)

    return f"🧬 {stats.summary()}" if stats else ""


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _on_refresh_system() -> str:
    return format_system_summary()


def _apply_expression_mode(mode: str):
    """Al elegir un Modo, rellena los controles con los valores recomendados."""
    preset = config.EXPRESSION_PRESETS.get(mode, config.EXPRESSION_PRESETS[config.EXPR_STANDARD])
    eng = preset.get("engine", config.ENGINE_INSIGHTFACE)
    return (
        eng,
        preset.get("ff_swapper_model", "inswapper_128"),
        preset.get("ff_pixel_boost", "256x256"),
        preset["enhancer_model"],
        preset["enhancer_blend"],
        preset.get("codeformer_fidelity", 0.7),
        preset["mask_mode"],
        preset["eye_preservation"],
        preset["mouth_detail"],
        preset["color_match"],
        preset["temporal_alpha"],
        preset["motion_adaptive"],
        preset["two_pass_temporal"],
        preset["reference_count"],
        preset.get("ram_mode", config.RAM_BALANCED),
        _recommendation(mode, eng),
    )


def _recommendation(mode: str, engine: str) -> str:
    """Recomendación automática según el modo y el motor elegidos (para la UI)."""
    tips = {
        config.EXPR_MUSIC_VIDEO: (
            "🎤 **Videos musicales** — ✅ **FaceFusion** con **inswapper_128** (el más **estable** "
            "en mucho movimiento: no se 'mueve'/desencaja) + **CodeFormer 512** para la nitidez + "
            "**post-procesado agresivo de boca/dientes** + **2 pasadas** + **RAM al máximo** + "
            "**6 referencias**. Sube **4–6 fotos** (frontal, 3/4 y perfil; boca abierta y cerrada). "
            "¿Querés más identidad/detalle? Probá los modelos de 256 px con **🔬 Comparar modelos** "
            "(transfieren la forma de cara: más parecido, pero pueden moverse más en perfiles)."
        ),
        config.EXPR_HIGH_EXPRESSION: (
            "😮 **Alta expresión** — FaceFusion + realce fuerte de **boca y ojos**. Ideal para "
            "primeros planos muy expresivos."
        ),
        config.EXPR_STANDARD: (
            "**Uso general** — **InsightFace** es más rápido y suele bastar. Cambia a **FaceFusion** "
            "si necesitas más calidad en **boca abierta o perfiles**."
        ),
    }
    base = tips.get(mode, "")
    if engine == config.ENGINE_FACEFUSION:
        extra = " ⏳ *FaceFusion prioriza calidad: más lento y más VRAM. Se instala solo la 1ª vez.*"
    else:
        extra = " ⚡ *InsightFace: rápido y menos VRAM.*"
    return f"### 💡 Recomendación\n{base}{extra}"


def _on_engine_change(engine: str, mode: str):
    """Al cambiar el motor: actualiza la recomendación y activa 2 pasadas con FaceFusion."""
    rec = _recommendation(mode, engine)
    two_pass = gr.update(value=True) if engine == config.ENGINE_FACEFUSION else gr.update()
    return rec, two_pass


def _memory_panel(engine: str, ram_mode: str, memory_mode: str, force_cpu: bool) -> str:
    """Métricas de memoria estimadas (se actualizan al cambiar motor/perfil)."""
    from ..core.memory_manager import MemoryManager

    s = config.Settings(
        engine=engine, ram_mode=ram_mode, memory_mode=memory_mode, force_cpu=bool(force_cpu)
    )
    try:
        return MemoryManager(s).format_metrics_md()
    except Exception:
        return ""


def _on_preview(source_files, video_path, n_preview, *control_values, progress=gr.Progress()):
    settings = _build_settings(*control_values)
    try:
        progress(0.02, desc="Cargando modelos…")
        pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.4, desc=m))
        src = _prepare(pipeline, source_files, video_path)

        def cb(frac, msg=""):
            progress(0.4 + frac * 0.6, desc=msg)

        results = pipeline.preview(video_path, n_frames=int(n_preview), progress=cb)
        return results, f"✅ Previsualización lista. {src}\nAjusta y vuelve a previsualizar si hace falta."
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Error en la previsualización")
        raise gr.Error(f"Error al previsualizar: {exc}")


def _on_compare(source_files, video_path, models, n_cmp, *control_values, progress=gr.Progress()):
    """A/B: corre varios modelos de swap sobre los mismos frames del material del usuario."""
    settings = _build_settings(*control_values)
    if not source_files:
        raise gr.Error("Subí al menos una imagen fuente (la cara a aplicar).")
    if not video_path:
        raise gr.Error("Subí un video objetivo.")
    models = [m for m in (models or [])]
    if len(models) < 2:
        raise gr.Error("Elegí al menos 2 modelos para comparar.")
    src_paths = [f if isinstance(f, str) else getattr(f, "name", None) for f in source_files]
    src_paths = [p for p in src_paths if p]

    from ..core.compare import compare_models
    try:
        items = compare_models(
            src_paths, video_path, models, settings, n_frames=int(n_cmp),
            progress=lambda f, m="": progress(f, desc=m),
        )
    except ValueError as exc:
        raise gr.Error(str(exc))
    except Exception as exc:  # pragma: no cover
        log.exception("Error en la comparación de modelos")
        raise gr.Error(f"Error al comparar: {exc}")
    msg = (f"✅ Comparados **{len(models)}** modelos × {int(n_cmp)} frames. Elegí a ojo el que "
           "mejor mantenga la **identidad** y el detalle, y ponelo arriba en *Modelo de swap de "
           "FaceFusion* (o cambiá el preset). El mejor depende de tu cara.")
    return items, msg


def _on_process(source_files, video_path, *control_values, progress=gr.Progress()):
    settings = _build_settings(*control_values)
    try:
        progress(0.0, desc="Cargando modelos…")
        pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.15, desc=m))
        src = _prepare(pipeline, source_files, video_path)

        def cb(frac, msg=""):
            progress(0.15 + frac * 0.85, desc=msg)

        out_path = pipeline.process_video(video_path, progress=cb)
        return out_path, out_path, f"✅ ¡Vídeo procesado! {src}  ·  Descárgalo abajo."
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Error al procesar el vídeo")
        raise gr.Error(f"Error al procesar: {exc}")


def _queue_duration_seconds(video_path: str) -> float:
    """Duración del video en segundos (para ordenar la cola). Los ilegibles van al final."""
    try:
        info = videoutil.probe(video_path)
        return info.frame_count / (info.fps or 25.0)
    except Exception:
        return float("inf")


def _on_process_queue(source_files, video_queue, *control_values, progress=gr.Progress()):
    """Procesa una COLA de videos con el MISMO set de imágenes fuente.

    - Los modelos se cargan una sola vez y se reutilizan (pipeline cacheado).
    - La cola se ordena del video MÁS CORTO al MÁS LARGO antes de empezar.
    - Cada resultado aparece para descargar EN CUANTO termina (no al final).
    - Si un video falla, se manda al final y se reintenta (hasta QUEUE_MAX_ATTEMPTS).

    Es un *generator*: va emitiendo (resultados, estado) tras cada video.
    """
    settings = _build_settings(*control_values)
    if not source_files:
        raise gr.Error("Sube al menos una imagen fuente (la cara a aplicar).")
    videos = [v if isinstance(v, str) else getattr(v, "name", None) for v in (video_queue or [])]
    videos = [v for v in videos if v]
    if not videos:
        raise gr.Error("Agrega al menos un video a la cola.")

    progress(0.0, desc="Cargando modelos…")
    pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.06, desc=m))

    images = []
    for f in source_files:
        path = f if isinstance(f, str) else getattr(f, "name", None)
        img = _read_image(path) if path else None
        if img is not None:
            images.append(img)
    if not images:
        raise gr.Error("No se pudieron leer las imágenes fuente.")
    try:
        stats = pipeline.prepare_source(images)
    except ValueError as exc:
        raise gr.Error(str(exc))

    # Ordenar la cola del MÁS CORTO al MÁS LARGO: las victorias rápidas salen primero.
    videos = sorted(videos, key=_queue_duration_seconds)

    n = len(videos)
    work = deque((v, 1) for v in videos)   # (ruta, nº de intento)
    outputs: list = []
    failed: list = []
    yield outputs, f"▶️ Cola de **{n}** videos (de más corto a más largo). Procesando…"

    while work:
        vpath, attempt = work.popleft()
        # En modo "por referencia", la referencia se toma de CADA video.
        if pipeline.settings.face_selector == config.FACE_SELECTOR_REFERENCE:
            info = videoutil.probe(vpath)
            mid = videoutil.get_frames_at(vpath, [info.frame_count // 2])
            if mid:
                pipeline.set_reference_from_frame(mid[0], pipeline.settings.reference_face_index)

        name = Path(vpath).name
        done = len(outputs) + len(failed)

        def cb(frac, msg="", _done=done, _name=name, _att=attempt):
            overall = 0.06 + 0.94 * (_done + frac) / n
            tag = f" (intento {_att})" if _att > 1 else ""
            progress(overall, desc=f"[{_done + 1}/{n}] {_name}{tag} · {msg}")

        try:
            out_name = f"{len(outputs) + 1:02d}_{Path(vpath).stem}_swap.mp4"
            out_path = str(config.OUTPUTS_DIR / out_name)
            outputs.append(pipeline.process_video(vpath, output_path=out_path, progress=cb))
            # Resultado disponible para descargar EN CUANTO termina este video.
            yield outputs, f"✅ {len(outputs)}/{n} listo · acabó **{name}**. Sigo con el resto…"
        except Exception:  # pragma: no cover
            log.exception("Falló %s (intento %d).", vpath, attempt)
            if attempt < QUEUE_MAX_ATTEMPTS:
                work.append((vpath, attempt + 1))   # al FINAL de la cola, para reintentar luego
                yield outputs, (f"⚠️ Falló **{name}**; lo mando al final de la cola para "
                                f"reintentarlo más tarde (intento {attempt + 1}/{QUEUE_MAX_ATTEMPTS})…")
            else:
                failed.append(name)
                yield outputs, f"❌ **{name}** falló {QUEUE_MAX_ATTEMPTS} veces; lo descarto. Sigo…"

    if not outputs:
        raise gr.Error("Ningún video de la cola se pudo procesar.")
    src = f"🧬 {stats.summary()}" if stats else ""
    tail = f" · descartados ({len(failed)}): {', '.join(failed)}" if failed else ""
    yield outputs, (f"✅ Cola completa: **{len(outputs)}/{n}** videos listos "
                    f"(más corto → más largo){tail}. {src}\nDescárgalos abajo.")


def _on_split_video(video, progress=gr.Progress()):
    """Corta un video en 5 partes IGUALES y las guarda en la carpeta ``chunks/``.

    Usa FFmpeg (re-encode preciso, conserva el audio). Las partes quedan tanto
    para descargar como en la carpeta ``chunks/`` dentro del proyecto.
    """
    import subprocess

    from ..utils.system import ffmpeg_path

    n_parts = 5
    if not video:
        raise gr.Error("Sube un video para cortar.")
    vpath = video if isinstance(video, str) else getattr(video, "name", None)
    if not vpath:
        raise gr.Error("No se pudo leer el video.")
    ff = ffmpeg_path()
    if not ff:
        raise gr.Error("FFmpeg no disponible.")

    info = videoutil.probe(vpath)
    total = info.frame_count / (info.fps or 25.0)
    if total <= 0:
        raise gr.Error("No se pudo determinar la duración del video.")
    part = total / n_parts

    chunks_dir = config.PROJECT_ROOT / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(vpath).stem

    outputs: list = []
    for i in range(n_parts):
        progress(i / n_parts, desc=f"Cortando parte {i + 1}/{n_parts}…")
        out = str(chunks_dir / f"{stem}_part{i + 1:02d}.mp4")
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-i", vpath, "-ss", f"{i * part:.3f}"]
        if i < n_parts - 1:                       # la última parte va hasta el final
            cmd += ["-t", f"{part:.3f}"]
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-c:a", "aac", "-movflags", "+faststart", out]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            outputs.append(out)
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or b"").decode("utf-8", "ignore")[-300:]
            raise gr.Error(f"FFmpeg falló al cortar la parte {i + 1}: {err}")

    progress(1.0, desc="Listo")
    return outputs, (f"✅ Video cortado en **{n_parts}** partes iguales → carpeta "
                     f"**chunks/** (`{chunks_dir}`).")


# ---------------------------------------------------------------------------
# Tema visual (handoff de diseño): "pro creative tool" oscuro + cyan
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ===== Fuser — tema oscuro + acento cyan (#00d4ff), fuente Inter ===== */
.gradio-container, .gradio-container.dark, :root {
  --body-background-fill: #0f1419;
  --body-background-fill-dark: #0f1419;
  --background-fill-primary: #1a1f2e;
  --background-fill-secondary: #0f1419;
  --block-background-fill: #1a1f2e;
  --block-border-color: #2a3142;
  --block-label-text-color: #00d4ff;
  --block-title-text-color: #00d4ff;
  --border-color-primary: #2a3142;
  --border-color-accent: #00d4ff;
  --body-text-color: #e0e0e0;
  --body-text-color-subdued: #aaaaaa;
  --color-accent: #00d4ff;
  --color-accent-soft: rgba(0, 212, 255, 0.12);
  --link-text-color: #00d4ff;
  --input-background-fill: #0f1419;
  --input-border-color: #2a3142;
  --button-primary-background-fill: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
  --button-primary-background-fill-hover: linear-gradient(135deg, #2ee0ff 0%, #00a8e0 100%);
  --button-primary-text-color: #06121a;
  --button-secondary-background-fill: #2a3142;
  --button-secondary-text-color: #e0e0e0;
  --slider-color: #00d4ff;
  --accordion-text-color: #00d4ff;
}
.gradio-container {
  background: #0f1419 !important;
  color: #e0e0e0 !important;
  max-width: 100% !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
/* Paneles / bloques */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .gr-group {
  background: #1a1f2e !important;
  border: 1px solid #2a3142 !important;
  border-radius: 8px !important;
}
/* Etiquetas de componentes y títulos de acordeón -> cyan */
.gradio-container .label-wrap span,
.gradio-container span[data-testid="block-info"],
.gradio-container .block > label > span {
  color: #cfe9f5 !important;
}
.gradio-container .label-wrap > span:first-child,
.gradio-container .accordion .label-wrap span {
  color: #00d4ff !important;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 600;
}
/* Botón primario: gradiente cyan */
.gradio-container button.primary,
.gradio-container .primary.svelte-cmf5ev,
.gradio-container button[variant="primary"] {
  background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
  color: #06121a !important;
  border: none !important;
  font-weight: 700 !important;
}
.gradio-container button.primary:hover {
  box-shadow: 0 8px 16px rgba(0, 212, 255, 0.25) !important;
}
/* Botón secundario */
.gradio-container button.secondary {
  background: #2a3142 !important;
  color: #e0e0e0 !important;
  border: 1px solid #3a4152 !important;
}
/* Zonas de carga (drag & drop) con borde discontinuo cyan al pasar */
.gradio-container .file-preview, .gradio-container .image-container,
.gradio-container [data-testid="block-label"] { color: #00d4ff !important; }
.gradio-container .upload-container, .gradio-container .file_preview,
.gradio-container .wrap.svelte-12ioyct {
  border: 2px dashed #2a3142 !important;
  border-radius: 8px !important;
}
.gradio-container .upload-container:hover {
  border-color: #00d4ff !important;
  background: rgba(0, 212, 255, 0.05) !important;
}
/* Sliders */
.gradio-container input[type=range] { accent-color: #00d4ff; }
/* Cabecera Fuser */
.fuser-header { border-bottom: 1px solid #2a3142; padding-bottom: 8px; margin-bottom: 4px; }
.fuser-header h1 { color: #fff !important; font-weight: 700; letter-spacing: -0.5px; }
.fuser-header .ver { color: #888; font-size: 0.6em; font-weight: 400; }
/* Acordeones: cabecera con look de panel */
.gradio-container .accordion { background: #1a1f2e !important; border: 1px solid #2a3142 !important; }
/* Galería de previsualización */
.gradio-container .gallery, .gradio-container .grid-wrap { background: #0f1419 !important; }
/* Markdown de recomendación / sistema: caja sutil */
.fuser-soft { background: #0f1419 !important; border: 1px solid #2a3142 !important;
  border-radius: 6px !important; padding: 4px 12px !important; }
"""


# ---------------------------------------------------------------------------
# Construcción de la interfaz
# ---------------------------------------------------------------------------
HEADER_MD = f"""
<div class="fuser-header">

# 🎭 {__app_name__} <span class="ver">v{__version__}</span>

**Face swap de vídeo 100% local** · optimizado para 8 GB VRAM + 40 GB RAM · afinado para caras cantando.
⚠️ Úsala solo con **consentimiento**; no para suplantar identidades ni desinformar.
</div>
"""

REFERENCE_TIP = """
**💡 Mejores resultados:** sube **3–5 fotos de la MISMA persona** (frontal, 3/4 y perfil; con
boca abierta y cerrada). Nítidas, bien iluminadas, sin gafas de sol ni manos/micros tapando la cara.
Más ángulos = más consistencia con la cabeza en movimiento (no cuesta VRAM extra).
"""


def build_interface() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Courier New", "monospace"],
    )
    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title=f"{__app_name__} · Video Face Swap") as demo:
        # ===== Cabecera =====
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(HEADER_MD)
            with gr.Column(scale=2):
                system_md = gr.Markdown(format_system_summary(), elem_classes="fuser-soft")
                refresh_btn = gr.Button("🔄 Actualizar estado del sistema", size="sm")

        # ===== Layout principal de 3 columnas =====
        with gr.Row(equal_height=False):
            # --------- COLUMNA IZQUIERDA: entradas ---------
            with gr.Column(scale=1, min_width=320):
                with gr.Group():
                    gr.Markdown("#### 🙂 Source Face")
                    source_files = gr.Files(
                        label="Imagen(es) fuente (la cara a aplicar)",
                        file_count="multiple", file_types=["image"], type="filepath",
                    )
                    gr.Markdown(REFERENCE_TIP, elem_classes="fuser-soft")
                with gr.Group():
                    gr.Markdown("#### 🎬 Target Video")
                    target_video = gr.Video(label="Vídeo objetivo")
                with gr.Group():
                    gr.Markdown("#### 🎤 Preset Mode")
                    expression_mode = gr.Dropdown(
                        choices=list(config.EXPRESSION_MODE_LABELS.items()),
                        value=config.EXPR_STANDARD,
                        label="Modo (ajusta la calidad automáticamente)",
                        info="Un clic configura motor, enhancer, máscaras, ojos/boca, 2 pasadas y RAM. "
                             "'Videos musicales' = el mejor preajuste para caras cantando.",
                    )
                    recommendation_md = gr.Markdown(
                        _recommendation(config.EXPR_STANDARD, config.ENGINE_INSIGHTFACE),
                        elem_classes="fuser-soft",
                    )

            # --------- COLUMNA CENTRAL: preview + resultado ---------
            with gr.Column(scale=1, min_width=340):
                with gr.Group():
                    gr.Markdown("#### 👁️ Preview")
                    n_preview = gr.Slider(
                        2, 12, value=6, step=1, label="Frames de previsualización",
                        info="Cuántos frames clave procesar para ver el resultado antes del vídeo "
                             "completo (incluye uno con boca abierta y uno de perfil).",
                    )
                    preview_btn = gr.Button("👁️ Previsualizar frames clave", variant="secondary")
                    preview_gallery = gr.Gallery(label="Previsualización", columns=3, object_fit="contain")
                    status_md = gr.Markdown("", elem_classes="fuser-soft")
                with gr.Group():
                    gr.Markdown("#### 🎉 Resultado")
                    output_video = gr.Video(label="Vídeo resultado")
                    output_file = gr.File(label="⬇️ Descargar resultado")
                gr.Markdown(
                    "🔬 *¿No sabés qué modelo usa mejor tu cara? Probá la pestaña "
                    "**Comparar modelos** abajo.*  ·  ⏭️✂️ *Cola y cortar video: pestaña "
                    "**Más herramientas**.*",
                    elem_classes="fuser-soft",
                )

            # --------- COLUMNA DERECHA: calidad y controles ---------
            with gr.Column(scale=1, min_width=340):
                with gr.Group():
                    gr.Markdown("#### ✨ Quality")
                    engine = gr.Radio(
                        choices=list(config.ENGINE_LABELS.items()),
                        value=config.ENGINE_INSIGHTFACE,
                        label="🧠 Motor de Face Swap",
                        info="Rápido (InsightFace) ↔ Alta calidad (FaceFusion).",
                    )
                    with gr.Row():
                        swapper_model = gr.Dropdown(
                            choices=config.SWAPPER_CHOICES, value="inswapper_128", label="Modelo de swap",
                            info="Modelo que reemplaza la identidad (InsightFace). inswapper_128 = más compatible.",
                        )
                        enhancer_model = gr.Dropdown(
                            choices=config.ENHANCER_CHOICES, value="gfpgan_1.4",
                            label="Enhancer (restaurador)",
                            info="GFPGAN = natural y rápido; CodeFormer = más nítido (mejor en dientes).",
                        )
                    with gr.Row():
                        enhancer_blend = gr.Slider(
                            0.0, 1.0, value=0.8, step=0.05, label="Fuerza del enhancer",
                            info="Cuánto se mezcla el realce (0 = nada, 1 = máximo).",
                        )
                        codeformer_fidelity = gr.Slider(
                            0.0, 1.0, value=0.7, step=0.05, label="Fidelidad CodeFormer",
                            info="Solo CodeFormer: 0 = más detalle/nítido, 1 = más fiel al original.",
                        )

                with gr.Accordion("👁️ Ajuste fino: ojos / boca / máscara (opcional)", open=False):
                    with gr.Row():
                        eye_preservation = gr.Slider(
                            0.0, 1.0, value=0.4, step=0.05, label="👁️ Preservación de ojos",
                            info="Realza los ojos para que no queden 'muertos'. Sube si la mirada pierde vida.",
                        )
                        mouth_detail = gr.Slider(
                            0.0, 1.0, value=0.4, step=0.05, label="👄 Detalle de boca/dientes",
                            info="Realza dientes/interior de la boca. Actúa más fuerte con la boca abierta.",
                        )
                    with gr.Row():
                        skin_detail = gr.Slider(
                            0.0, 1.0, value=0.35, step=0.05, label="🧴 Textura de piel (anti-plástico)",
                            info="Reinyecta la textura/poros del video original sobre la cara swapeada. "
                                 "Sube si la piel se ve cerosa/plástica; baja si aparece grano de más.",
                        )
                    with gr.Row():
                        mouth_enhancer = gr.Checkbox(
                            value=True, label="🦷 Enhancer localizado de boca (CodeFormer, FaceFusion)",
                            info="2.º pase de CodeFormer SOLO en la boca abierta (FaceFusion). Dientes más nítidos.",
                        )
                        color_match = gr.Checkbox(
                            value=False, label="Igualar color al original (iluminación)",
                            info="Adapta el tono de la cara nueva al del vídeo. Útil con luces de escenario.",
                        )
                    with gr.Row():
                        mask_mode = gr.Dropdown(
                            choices=list(config.MASK_MODE_LABELS.items()), value=config.MASK_HULL,
                            label="🎭 Tipo de máscara",
                            info="'Contorno' sigue el rostro real (ideal perfiles); 'Rectángulo' es lo más básico.",
                        )
                        face_opacity = gr.Slider(
                            0.1, 1.0, value=1.0, step=0.05, label="Fuerza del swap (opacidad)",
                            info="Intensidad del intercambio. <1 deja ver algo de la cara original.",
                        )
                    with gr.Row():
                        mask_blur = gr.Slider(
                            0.0, 0.8, value=0.25, step=0.05, label="Suavizado del borde",
                            info="Difumina el borde de la máscara. Súbelo si ves una costura marcada.",
                        )
                        mask_padding = gr.Slider(
                            0.0, 0.4, value=0.0, step=0.02, label="Recorte interior",
                            info="Encoge la máscara hacia dentro para no invadir pelo/frente/orejas.",
                        )

                with gr.Accordion("⚙️ FaceFusion avanzado (resolución interna del swap)", open=False):
                    gr.Markdown(config.ENGINE_INFO_MD)
                    with gr.Row():
                        ff_swapper_model = gr.Dropdown(
                            choices=config.FF_SWAPPER_CHOICES, value="inswapper_128",
                            label="Modelo de swap de FaceFusion",
                            info="inswapper_128 = el más ESTABLE (no se 'mueve'). Para más identidad "
                                 "con OJOS nítidos → ghost_3_256. simswap da un look suave (ojos "
                                 "borrosos). Los de 256 px transfieren la forma → se mueven más en perfiles.",
                        )
                        ff_pixel_boost = gr.Dropdown(
                            choices=config.FF_PIXEL_BOOST_CHOICES, value="256x256",
                            label="Pixel boost (256/512 = más calidad, más VRAM)",
                            info="Resolución interna del swap. Más alto = dientes/ojos más nítidos, más VRAM.",
                        )
                    with gr.Row():
                        use_mouth_pixel_boost = gr.Checkbox(
                            value=True, label="Pixel boost localizado de boca (512)",
                            info="Reprocesa la boca con CodeFormer a 512 cuando está abierta. Dientes más definidos.",
                        )
                        mouth_enhancement_strength = gr.Slider(
                            0.0, 2.0, value=1.0, step=0.1, label="Fuerza del enhancer de boca",
                            info="Multiplica la fuerza del realce localizado de boca (1.0 = normal, >1 = agresivo).",
                        )
                        profile_blending_strength = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05, label="Blending en perfiles",
                            info="En caras de lado recupera el borde original (mandíbula/oreja) para evitar deformación.",
                        )

                with gr.Accordion("👥 Selección de caras y multi-referencia", open=False):
                    with gr.Row():
                        face_selector = gr.Dropdown(
                            choices=list(config.FACE_SELECTOR_LABELS.items()),
                            value=config.FACE_SELECTOR_ALL, label="¿A qué caras aplicar el swap?",
                            info="Todas, la más grande, una persona (por referencia) o por posición.",
                        )
                        reference_count = gr.Dropdown(
                            choices=config.REFERENCE_COUNT_CHOICES, value=0,
                            label="Nº de imágenes de referencia",
                            info="Cuántas fotos de origen combinar. 4–6 = más consistencia. No cuesta VRAM extra.",
                        )
                    with gr.Row():
                        reference_index = gr.Number(
                            value=0, precision=0, label="Índice de cara (izq→der)",
                            info="Para 'por posición'/'por referencia': qué cara del vídeo (0 = la primera).",
                        )
                        reference_distance = gr.Slider(
                            0.5, 2.0, value=1.2, step=0.05, label="Tolerancia de referencia",
                            info="En modo 'por referencia': qué tan parecida debe ser para considerarla la misma persona.",
                        )

                with gr.Accordion("🎞️ Temporal (estabilidad / movimiento)", open=False):
                    with gr.Row():
                        temporal_smoothing = gr.Checkbox(
                            value=True, label="Suavizado temporal",
                            info="Reduce el temblor de la cara entre frames.",
                        )
                        motion_adaptive = gr.Checkbox(
                            value=True, label="Adaptativo al movimiento (sin lag)",
                            info="Suaviza el temblor pero reacciona al instante (la boca al cantar no se arrastra).",
                        )
                    with gr.Row():
                        temporal_alpha = gr.Slider(
                            0.0, 0.9, value=0.55, step=0.05, label="Intensidad del suavizado",
                            info="Más alto = más estable, pero puede dar 'lag' en movimientos rápidos.",
                        )
                        two_pass_temporal = gr.Checkbox(
                            value=False, label="2 pasadas (máxima estabilidad, usa RAM)",
                            info="Analiza un tramo en RAM y estabiliza con ventana centrada (sin lag). Más calidad.",
                        )

                with gr.Accordion("🧠 Memoria y rendimiento", open=False):
                    with gr.Row():
                        memory_mode = gr.Dropdown(
                            choices=list(config.MEMORY_MODE_LABELS.items()),
                            value=config.MODE_BALANCED, label="Modo de memoria (VRAM)",
                            info="Baja a 'Bajo VRAM' / 'VRAM mínima' si da error de memoria de GPU.",
                        )
                        ram_mode = gr.Dropdown(
                            choices=list(config.RAM_MODE_LABELS.items()),
                            value=config.RAM_BALANCED, label="Uso de RAM (buffers / 2 pasadas)",
                            info="'Máximo' aprovecha 32 GB+ (la GPU casi nunca espera).",
                        )
                        gpu_mem_limit = gr.Slider(
                            0.0, 12.0, value=0.0, step=0.5, label="Límite VRAM/sesión (GB, 0=auto)",
                            info="Tope de VRAM por modelo. 0 = automático según el modo.",
                        )
                    with gr.Row():
                        processing_resolution = gr.Dropdown(
                            choices=[("Nativa (máxima calidad)", 0), ("1440p", 1440), ("1080p", 1080),
                                     ("720p (rápido)", 720), ("512p (muy rápido)", 512)],
                            value=0, label="Resolución de procesamiento",
                            info="Menor = más rápido y menos memoria, pero menos detalle.",
                        )
                        force_cpu = gr.Checkbox(
                            value=False, label="Forzar CPU (sin GPU)",
                            info="Procesa sin GPU. Muy lento; solo para probar la UI.",
                        )
                        keep_audio = gr.Checkbox(value=True, label="Conservar audio")
                        keep_fps = gr.Checkbox(value=True, label="Conservar FPS")
                        output_quality = gr.Slider(
                            12, 30, value=18, step=1, label="Calidad de salida (CRF)",
                            info="H.264: menor = mejor calidad y archivo más pesado (18 es buen punto).",
                        )
                    mem_info_md = gr.Markdown(
                        _memory_panel(config.ENGINE_INSIGHTFACE, config.RAM_BALANCED,
                                      config.MODE_BALANCED, False),
                        elem_classes="fuser-soft",
                    )

                with gr.Group():
                    gr.Markdown("#### 🚀 Process")
                    qc_second_pass = gr.Checkbox(
                        value=False, label="🔍 2ª pasada: detectar y corregir defectos",
                        info="Al terminar el vídeo, hace un repaso: detecta frames defectuosos "
                             "(cara sin swapear, borrosa, identidad rara o salto) y los CORRIGE con "
                             "el MISMO modelo (re-swap con detección agresiva o relleno desde vecinos "
                             "buenos). Suma tiempo al final; usa RAM (mejor en clips cortos).",
                    )
                    qc_sensitivity = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05, label="Sensibilidad de la 2ª pasada",
                        info="0 = solo defectos claros · 1 = agresivo (marca y corrige más frames).",
                    )
                    process_btn = gr.Button("🚀 Procesar vídeo completo", variant="primary")

        # ===== Más modos: Imagen→Vídeo · comparar modelos · herramientas =====
        gr.Markdown("### 🧭 Más modos y herramientas")
        with gr.Tabs():
            if build_i2v_tab is not None:
                with gr.Tab("🎞️ Imagen → Vídeo (Wan 2.2)"):
                    build_i2v_tab()
            with gr.Tab("🔬 Comparar modelos"):
                gr.Markdown(
                    "Probá varios modelos de swap sobre **tu** cara y **tu** video, en los mismos "
                    "frames clave. El mejor **depende de tu cara** (lo medimos: *ghost_3* suele ganar "
                    "en identidad, pero a veces *hififace*/*simswap* quedan mejor). Usa las mismas "
                    "**imágenes fuente** y el mismo **vídeo** de la izquierda.\n\n"
                    "⏱️ *Cada modelo corre en su propio proceso (por cómo DirectML retiene la VRAM), "
                    "así que tarda ~10–20 s por modelo.*"
                )
                cmp_models = gr.CheckboxGroup(
                    choices=config.FF_SWAPPER_CHOICES,
                    value=["ghost_3_256", "hififace_unofficial_256", "simswap_256", "inswapper_128"],
                    label="Modelos a comparar (elegí 2 o más)",
                )
                with gr.Row():
                    cmp_frames = gr.Slider(2, 6, value=3, step=1, label="Frames clave por modelo")
                    cmp_btn = gr.Button("🔬 Comparar en mi material", variant="primary")
                cmp_gallery = gr.Gallery(
                    label="Comparación recortada a la cara (ORIGINAL · modelos)",
                    columns=5, object_fit="contain", height=440,
                )
                cmp_status = gr.Markdown("", elem_classes="fuser-soft")
            with gr.Tab("🧰 Más herramientas"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### ⏭️ Cola de trabajos (mismo set de fuentes)")
                        video_queue = gr.Files(
                            label="Videos cortos a procesar en cola (uno tras otro)",
                            file_count="multiple", file_types=["video"], type="filepath",
                        )
                        queue_btn = gr.Button("⏭️ Procesar cola", variant="primary")
                        queue_status = gr.Markdown("", elem_classes="fuser-soft")
                        queue_results = gr.Files(label="⬇️ Resultados de la cola (descargar)")
                    with gr.Column():
                        gr.Markdown("#### ✂️ Cortar video en partes")
                        split_video = gr.Video(label="Video a cortar")
                        split_btn = gr.Button("✂️ Cortar en 5 partes iguales", variant="secondary")
                        split_status = gr.Markdown("", elem_classes="fuser-soft")
                        split_results = gr.Files(
                            label="⬇️ Partes (también quedan en la carpeta chunks/)")

        # ----- Orden EXACTO = firma de _build_settings -----------------------
        control_inputs = [
            engine, ff_swapper_model, ff_pixel_boost,
            use_mouth_pixel_boost, mouth_enhancement_strength, profile_blending_strength,
            swapper_model, enhancer_model, enhancer_blend, codeformer_fidelity,
            expression_mode,
            face_selector, reference_index, reference_distance, reference_count,
            face_opacity, mask_mode, mask_blur, mask_padding, eye_preservation, mouth_detail,
            mouth_enhancer, color_match, processing_resolution,
            temporal_smoothing, temporal_alpha, motion_adaptive, two_pass_temporal,
            memory_mode, gpu_mem_limit, force_cpu, ram_mode,
            keep_audio, keep_fps, output_quality,
            skin_detail,
            qc_second_pass, qc_sensitivity,
        ]

        # ----- Wiring (sin cambios) -----------------------------------------
        refresh_btn.click(_on_refresh_system, inputs=None, outputs=system_md)

        expression_mode.change(
            _apply_expression_mode,
            inputs=expression_mode,
            outputs=[
                engine, ff_swapper_model, ff_pixel_boost,
                enhancer_model, enhancer_blend, codeformer_fidelity, mask_mode,
                eye_preservation, mouth_detail, color_match, temporal_alpha,
                motion_adaptive, two_pass_temporal, reference_count, ram_mode,
                recommendation_md,
            ],
        )
        engine.change(
            _on_engine_change,
            inputs=[engine, expression_mode],
            outputs=[recommendation_md, two_pass_temporal],
        )
        _mem_inputs = [engine, ram_mode, memory_mode, force_cpu]
        for _comp in (engine, ram_mode, memory_mode, force_cpu):
            _comp.change(_memory_panel, inputs=_mem_inputs, outputs=mem_info_md)

        preview_btn.click(
            _on_preview,
            inputs=[source_files, target_video, n_preview, *control_inputs],
            outputs=[preview_gallery, status_md],
        )
        cmp_btn.click(
            _on_compare,
            inputs=[source_files, target_video, cmp_models, cmp_frames, *control_inputs],
            outputs=[cmp_gallery, cmp_status],
        )
        process_btn.click(
            _on_process,
            inputs=[source_files, target_video, *control_inputs],
            outputs=[output_video, output_file, status_md],
        )
        queue_btn.click(
            _on_process_queue,
            inputs=[source_files, video_queue, *control_inputs],
            outputs=[queue_results, queue_status],
        )
        split_btn.click(
            _on_split_video,
            inputs=[split_video],
            outputs=[split_results, split_status],
        )

    return demo
