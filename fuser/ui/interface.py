"""Interfaz Gradio de Fuser.

Optimizada para el caso de uso de **caras de mujeres en videos musicales**:
múltiples ángulos, boca muy abierta al cantar, expresiones intensas y mucho
movimiento. El **Modo** (arriba) ajusta de golpe enhancer, máscara, preservación
de ojos/boca, estabilidad temporal y nº de referencias recomendado.

El pipeline se cachea entre ejecuciones: solo se recargan modelos si cambian los
ajustes que afectan a modelos/memoria; el resto se aplica en caliente.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import gradio as gr
import numpy as np

from .. import __app_name__, __version__, config
from ..core.pipeline import SwapPipeline
from ..utils import video as videoutil
from ..utils.logging import get_logger
from ..utils.system import format_system_summary

log = get_logger(__name__)

_PIPELINE_CACHE: dict = {"pipeline": None, "signature": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_image(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _build_settings(
    engine, ff_swapper_model, ff_pixel_boost,
    swapper_model, enhancer_model, enhancer_blend, codeformer_fidelity,
    expression_mode,
    face_selector, reference_index, reference_distance, reference_count,
    face_opacity, mask_mode, mask_blur, mask_padding, eye_preservation, mouth_detail,
    color_match, processing_resolution,
    temporal_smoothing, temporal_alpha, motion_adaptive, two_pass_temporal,
    memory_mode, gpu_mem_limit, force_cpu, ram_boost,
    keep_audio, keep_fps, output_quality,
) -> config.Settings:
    return config.Settings(
        engine=engine,
        ff_swapper_model=ff_swapper_model,
        ff_pixel_boost=ff_pixel_boost,
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
        color_match=bool(color_match),
        processing_resolution=int(processing_resolution),
        temporal_smoothing=bool(temporal_smoothing),
        temporal_alpha=float(temporal_alpha),
        motion_adaptive=bool(motion_adaptive),
        two_pass_temporal=bool(two_pass_temporal),
        memory_mode=memory_mode,
        gpu_mem_limit_gb=float(gpu_mem_limit),
        force_cpu=bool(force_cpu),
        ram_boost=bool(ram_boost),
        keep_audio=bool(keep_audio),
        keep_fps=bool(keep_fps),
        output_quality=int(output_quality),
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
    return (
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
    )


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


# ---------------------------------------------------------------------------
# Construcción de la interfaz
# ---------------------------------------------------------------------------
DESCRIPTION = f"""
# 🎭 {__app_name__} · Face swap de vídeo local  <span style="font-size:0.6em">v{__version__}</span>

Face swap de vídeo de alta calidad **100% local**, optimizado para **8 GB de VRAM + 40 GB de RAM** y
afinado para **caras cantando en videos musicales** (múltiples ángulos, boca abierta, perfiles).

> ⚠️ **Uso responsable:** úsala solo con **consentimiento** de las personas involucradas.
> No la utilices para suplantar identidades, desinformar ni crear contenido engañoso.
"""

REFERENCE_TIP = """
**📸 Cómo subir buenas referencias (clave para videos musicales):**
sube **3–5 fotos de la MISMA persona** que cubran lo que aparece en el vídeo:
**frontal, 3/4 y perfil**, con **boca cerrada y abierta/sonriendo**. Fotos **nítidas, bien iluminadas**,
sin gafas de sol, sin manos/micros tapando la cara y sin filtros fuertes. Más ángulos = más consistencia
en cabeza en movimiento (no cuesta VRAM extra).
"""


def build_interface() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="slate")
    with gr.Blocks(theme=theme, title=f"{__app_name__} · Video Face Swap") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    source_files = gr.Files(
                        label="🙂 Imagen(es) fuente (la cara a aplicar)",
                        file_count="multiple", file_types=["image"], type="filepath",
                    )
                    target_video = gr.Video(label="🎬 Vídeo objetivo")
                gr.Markdown(REFERENCE_TIP)
            with gr.Column(scale=2):
                system_md = gr.Markdown(format_system_summary())
                refresh_btn = gr.Button("🔄 Actualizar estado del sistema", size="sm")

        # ----- Motor de face swap (toggle) ----------------------------------
        with gr.Row():
            engine = gr.Radio(
                choices=list(config.ENGINE_LABELS.items()),
                value=config.ENGINE_INSIGHTFACE,
                label="🧠 Motor de Face Swap",
                info="Cambia entre rápido (InsightFace) y alta calidad (FaceFusion).",
            )
        gr.Markdown(config.ENGINE_INFO_MD)

        # ----- Modo (caso de uso) -------------------------------------------
        with gr.Row():
            expression_mode = gr.Dropdown(
                choices=list(config.EXPRESSION_MODE_LABELS.items()),
                value=config.EXPR_STANDARD,
                label="🎚️ Modo (ajusta automáticamente la calidad para el caso de uso)",
            )

        # ----- Modelos y calidad --------------------------------------------
        with gr.Accordion("🎨 Modelos y calidad", open=True):
            with gr.Row():
                swapper_model = gr.Dropdown(
                    choices=config.SWAPPER_CHOICES, value="inswapper_128", label="Modelo de swap",
                )
                enhancer_model = gr.Dropdown(
                    choices=config.ENHANCER_CHOICES, value="gfpgan_1.4",
                    label="Enhancer (restaurador de cara)",
                )
            with gr.Row():
                enhancer_blend = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Fuerza del enhancer")
                codeformer_fidelity = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.05, label="Fidelidad CodeFormer (0=detalle, 1=fiel)",
                )
            with gr.Row():
                ff_swapper_model = gr.Dropdown(
                    choices=config.FF_SWAPPER_CHOICES, value="inswapper_128",
                    label="FaceFusion · modelo de swap (solo motor FaceFusion)",
                )
                ff_pixel_boost = gr.Dropdown(
                    choices=config.FF_PIXEL_BOOST_CHOICES, value="256x256",
                    label="FaceFusion · pixel boost (resolución del swap)",
                )

        # ----- Preservación de detalle --------------------------------------
        with gr.Accordion("✨ Preservación de ojos, boca y máscara", open=True):
            with gr.Row():
                eye_preservation = gr.Slider(
                    0.0, 1.0, value=0.4, step=0.05,
                    label="👁️ Preservación de ojos (nitidez/vida de la mirada)",
                )
                mouth_detail = gr.Slider(
                    0.0, 1.0, value=0.4, step=0.05,
                    label="👄 Detalle de boca/dientes (al cantar)",
                )
            with gr.Row():
                mask_mode = gr.Dropdown(
                    choices=list(config.MASK_MODE_LABELS.items()), value=config.MASK_HULL,
                    label="Tipo de máscara (contorno = mejor en perfiles)",
                )
                face_opacity = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Fuerza del swap (opacidad)")
                color_match = gr.Checkbox(value=False, label="Igualar color al original (iluminación)")
            with gr.Row():
                mask_blur = gr.Slider(0.0, 0.8, value=0.25, step=0.05, label="Suavizado del borde (máscara)")
                mask_padding = gr.Slider(0.0, 0.4, value=0.0, step=0.02, label="Recorte interior (máscara)")

        # ----- Selección de caras y referencias -----------------------------
        with gr.Accordion("👥 Selección de caras y multi-referencia", open=False):
            with gr.Row():
                face_selector = gr.Dropdown(
                    choices=list(config.FACE_SELECTOR_LABELS.items()),
                    value=config.FACE_SELECTOR_ALL, label="¿A qué caras aplicar el swap?",
                )
                reference_count = gr.Dropdown(
                    choices=config.REFERENCE_COUNT_CHOICES, value=0,
                    label="Nº de imágenes de referencia a usar",
                )
            with gr.Row():
                reference_index = gr.Number(value=0, precision=0, label="Índice de cara (izq→der)")
                reference_distance = gr.Slider(
                    0.5, 2.0, value=1.2, step=0.05,
                    label="Tolerancia de referencia (mayor = más permisivo)",
                )

        # ----- Estabilidad temporal -----------------------------------------
        with gr.Accordion("🎞️ Estabilidad temporal (movimiento)", open=False):
            with gr.Row():
                temporal_smoothing = gr.Checkbox(value=True, label="Suavizado temporal")
                temporal_alpha = gr.Slider(
                    0.0, 0.9, value=0.55, step=0.05, label="Intensidad del suavizado",
                )
            with gr.Row():
                motion_adaptive = gr.Checkbox(
                    value=True, label="Adaptativo al movimiento (sin lag en la boca)",
                )
                two_pass_temporal = gr.Checkbox(
                    value=False, label="2 pasadas: máxima estabilidad (usa RAM)",
                )

        # ----- Memoria y rendimiento ----------------------------------------
        with gr.Accordion("🧠 Memoria y rendimiento", open=False):
            with gr.Row():
                memory_mode = gr.Dropdown(
                    choices=list(config.MEMORY_MODE_LABELS.items()),
                    value=config.MODE_BALANCED, label="Modo de memoria",
                )
                gpu_mem_limit = gr.Slider(
                    0.0, 12.0, value=0.0, step=0.5, label="Límite VRAM/sesión (GB, 0=auto)",
                )
                processing_resolution = gr.Dropdown(
                    choices=[("Nativa (máxima calidad)", 0), ("1440p", 1440), ("1080p", 1080),
                             ("720p (rápido)", 720), ("512p (muy rápido)", 512)],
                    value=0, label="Resolución de procesamiento",
                )
            with gr.Row():
                ram_boost = gr.Checkbox(value=True, label="Usar más RAM (buffers grandes)")
                force_cpu = gr.Checkbox(value=False, label="Forzar CPU (sin GPU)")
                keep_audio = gr.Checkbox(value=True, label="Conservar audio")
                keep_fps = gr.Checkbox(value=True, label="Conservar FPS")
                output_quality = gr.Slider(12, 30, value=18, step=1, label="Calidad de salida (CRF)")

        # ----- Orden EXACTO = firma de _build_settings -----------------------
        control_inputs = [
            engine, ff_swapper_model, ff_pixel_boost,
            swapper_model, enhancer_model, enhancer_blend, codeformer_fidelity,
            expression_mode,
            face_selector, reference_index, reference_distance, reference_count,
            face_opacity, mask_mode, mask_blur, mask_padding, eye_preservation, mouth_detail,
            color_match, processing_resolution,
            temporal_smoothing, temporal_alpha, motion_adaptive, two_pass_temporal,
            memory_mode, gpu_mem_limit, force_cpu, ram_boost,
            keep_audio, keep_fps, output_quality,
        ]

        # ----- Acciones ------------------------------------------------------
        gr.Markdown("### ▶️ Acciones")
        with gr.Row():
            n_preview = gr.Slider(2, 12, value=6, step=1, label="Frames de previsualización")
            preview_btn = gr.Button("👁️ Previsualizar frames clave", variant="secondary")
            process_btn = gr.Button("🚀 Procesar vídeo completo", variant="primary")

        status_md = gr.Markdown("")
        preview_gallery = gr.Gallery(label="Previsualización", columns=3, object_fit="contain")
        with gr.Row():
            output_video = gr.Video(label="🎉 Resultado")
            output_file = gr.File(label="⬇️ Descargar resultado")

        # ----- Wiring --------------------------------------------------------
        refresh_btn.click(_on_refresh_system, inputs=None, outputs=system_md)

        expression_mode.change(
            _apply_expression_mode,
            inputs=expression_mode,
            outputs=[
                enhancer_model, enhancer_blend, codeformer_fidelity, mask_mode,
                eye_preservation, mouth_detail, color_match, temporal_alpha,
                motion_adaptive, two_pass_temporal, reference_count,
            ],
        )

        preview_btn.click(
            _on_preview,
            inputs=[source_files, target_video, n_preview, *control_inputs],
            outputs=[preview_gallery, status_md],
        )
        process_btn.click(
            _on_process,
            inputs=[source_files, target_video, *control_inputs],
            outputs=[output_video, output_file, status_md],
        )

        gr.Markdown(
            "💡 **Para videos musicales:** elige el modo **🎤 Videos musicales**, sube **3–5 fotos** "
            "en varios ángulos, **previsualiza** un par de frames (incluido uno con la boca abierta) y "
            "afina *Preservación de ojos* y *Detalle de boca*. En 8 GB usa **Equilibrado**; activa "
            "**2 pasadas** para máxima estabilidad si te sobra RAM."
        )

    return demo
