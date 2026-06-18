"""Interfaz Gradio de Fuser.

Diseño:
- Columna de entradas (fuente + vídeo) y panel de estado del sistema.
- Acordeones con controles agrupados: modelos/calidad, selección de caras,
  estabilidad temporal y memoria/rendimiento.
- Acciones: previsualizar frames clave y procesar el vídeo completo, ambas con
  barra de progreso real y ETA.

El pipeline se cachea entre ejecuciones: solo se reconstruye (recargando modelos)
cuando cambian los ajustes que afectan a los modelos o a la memoria. Los ajustes
"ligeros" (máscara, opacidad, selector...) se aplican sin recargar.
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

# Caché de un único pipeline (modelos cargados) entre ejecuciones de la UI.
_PIPELINE_CACHE: dict = {"pipeline": None, "signature": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _build_settings(
    swapper_model, enhancer_model, enhancer_blend, codeformer_fidelity,
    face_selector, reference_index, reference_distance,
    face_opacity, mask_blur, mask_padding, color_match, processing_resolution,
    temporal_smoothing, temporal_alpha,
    memory_mode, gpu_mem_limit, force_cpu,
    keep_audio, keep_fps, output_quality,
) -> config.Settings:
    return config.Settings(
        swapper_model=swapper_model,
        enhancer_model=enhancer_model,
        enhancer_blend=float(enhancer_blend),
        codeformer_fidelity=float(codeformer_fidelity),
        face_selector=face_selector,
        reference_face_index=int(reference_index),
        reference_distance=float(reference_distance),
        face_opacity=float(face_opacity),
        mask_blur=float(mask_blur),
        mask_padding=float(mask_padding),
        color_match=bool(color_match),
        processing_resolution=int(processing_resolution),
        temporal_smoothing=bool(temporal_smoothing),
        temporal_alpha=float(temporal_alpha),
        memory_mode=memory_mode,
        gpu_mem_limit_gb=float(gpu_mem_limit),
        force_cpu=bool(force_cpu),
        keep_audio=bool(keep_audio),
        keep_fps=bool(keep_fps),
        output_quality=int(output_quality),
    )


def _get_pipeline(settings: config.Settings, progress=None) -> SwapPipeline:
    """Devuelve un pipeline cargado, reutilizando la caché cuando es posible.

    Solo se recargan modelos si cambian los ajustes que afectan a modelos/memoria
    (``model_signature``); el resto se aplica en caliente con ``update_runtime``.
    """
    pipeline = SwapPipeline(settings)
    signature = pipeline.model_signature()

    cached = _PIPELINE_CACHE["pipeline"]
    if cached is not None and _PIPELINE_CACHE["signature"] == signature:
        cached.update_runtime(settings)  # se descarta el pipeline recién creado (sin modelos)
        return cached

    pipeline.load_models(progress=progress)
    _PIPELINE_CACHE["pipeline"] = pipeline
    _PIPELINE_CACHE["signature"] = signature
    return pipeline


def _prepare(pipeline: SwapPipeline, source_files, video_path) -> None:
    """Valida entradas y prepara la cara fuente (+ referencia si aplica)."""
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
        pipeline.prepare_source(images)
    except ValueError as exc:
        raise gr.Error(str(exc))

    # Modo "referencia": fija la cara de referencia desde un frame central.
    if pipeline.settings.face_selector == config.FACE_SELECTOR_REFERENCE:
        info = videoutil.probe(video_path)
        mid = videoutil.get_frames_at(video_path, [info.frame_count // 2])
        if mid:
            pipeline.set_reference_from_frame(mid[0], pipeline.settings.reference_face_index)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _on_refresh_system() -> str:
    return format_system_summary()


def _on_preview(
    source_files, video_path, n_preview, *control_values, progress=gr.Progress()
):
    settings = _build_settings(*control_values)
    try:
        progress(0.02, desc="Cargando modelos…")
        pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.4, desc=m))
        _prepare(pipeline, source_files, video_path)

        def cb(frac, msg=""):
            progress(0.4 + frac * 0.6, desc=msg)

        results = pipeline.preview(video_path, n_frames=int(n_preview), progress=cb)
        return results, "✅ Previsualización lista. Ajusta los controles y vuelve a previsualizar si hace falta."
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Error en la previsualización")
        raise gr.Error(f"Error al previsualizar: {exc}")


def _on_process(
    source_files, video_path, *control_values, progress=gr.Progress()
):
    settings = _build_settings(*control_values)
    try:
        progress(0.0, desc="Cargando modelos…")
        pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.15, desc=m))
        _prepare(pipeline, source_files, video_path)

        def cb(frac, msg=""):
            progress(0.15 + frac * 0.85, desc=msg)

        out_path = pipeline.process_video(video_path, progress=cb)
        return out_path, out_path, "✅ ¡Vídeo procesado! Descárgalo abajo."
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

Face swap de vídeo de alta calidad **100% local**, optimizado para **8 GB de VRAM + 40 GB de RAM**.
Sube una **cara fuente** y un **vídeo objetivo**, ajusta la calidad y procesa.

> ⚠️ **Uso responsable:** usa esta herramienta solo con consentimiento de las personas involucradas.
> No la utilices para suplantar identidades, desinformar ni crear contenido engañoso.
"""


def build_interface() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="slate")
    with gr.Blocks(theme=theme, title=f"{__app_name__} · Video Face Swap") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # ----- Entradas --------------------------------------------------
            with gr.Column(scale=3):
                with gr.Row():
                    source_files = gr.Files(
                        label="🙂 Imagen(es) fuente (la cara a aplicar)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    target_video = gr.Video(label="🎬 Vídeo objetivo")

            # ----- Estado del sistema ---------------------------------------
            with gr.Column(scale=2):
                system_md = gr.Markdown(format_system_summary())
                refresh_btn = gr.Button("🔄 Actualizar estado del sistema", size="sm")

        # ----- Controles avanzados ------------------------------------------
        with gr.Accordion("🎨 Modelos y calidad", open=True):
            with gr.Row():
                swapper_model = gr.Dropdown(
                    choices=config.SWAPPER_CHOICES, value="inswapper_128",
                    label="Modelo de swap",
                )
                enhancer_model = gr.Dropdown(
                    choices=config.ENHANCER_CHOICES, value="gfpgan_1.4",
                    label="Enhancer (restaurador de cara)",
                )
            with gr.Row():
                enhancer_blend = gr.Slider(
                    0.0, 1.0, value=0.8, step=0.05, label="Fuerza del enhancer",
                )
                codeformer_fidelity = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.05,
                    label="Fidelidad CodeFormer (0=detalle, 1=fiel)",
                )
            with gr.Row():
                face_opacity = gr.Slider(
                    0.1, 1.0, value=1.0, step=0.05, label="Fuerza del swap (opacidad)",
                )
                processing_resolution = gr.Dropdown(
                    choices=[("Nativa (máxima calidad)", 0), ("1440p", 1440),
                             ("1080p", 1080), ("720p (rápido / menos memoria)", 720),
                             ("512p (muy rápido)", 512)],
                    value=0, label="Resolución de procesamiento",
                )
            with gr.Row():
                mask_blur = gr.Slider(
                    0.0, 0.8, value=0.25, step=0.05, label="Suavizado del borde (máscara)",
                )
                mask_padding = gr.Slider(
                    0.0, 0.4, value=0.0, step=0.02, label="Recorte interior (máscara)",
                )
                color_match = gr.Checkbox(
                    value=False, label="Igualar color al original",
                )

        with gr.Accordion("👥 Selección de caras", open=False):
            with gr.Row():
                face_selector = gr.Dropdown(
                    choices=list(config.FACE_SELECTOR_LABELS.items()),
                    value=config.FACE_SELECTOR_ALL,
                    label="¿A qué caras aplicar el swap?",
                )
                reference_index = gr.Number(
                    value=0, precision=0, label="Índice de cara (izq→der)",
                )
                reference_distance = gr.Slider(
                    0.5, 2.0, value=1.2, step=0.05,
                    label="Tolerancia de referencia (mayor = más permisivo)",
                )

        with gr.Accordion("🎞️ Estabilidad temporal", open=False):
            with gr.Row():
                temporal_smoothing = gr.Checkbox(
                    value=True, label="Suavizado temporal (reduce el temblor entre frames)",
                )
                temporal_alpha = gr.Slider(
                    0.0, 0.9, value=0.55, step=0.05,
                    label="Intensidad del suavizado (alto = más estable, más 'lag')",
                )

        with gr.Accordion("🧠 Memoria y rendimiento", open=False):
            with gr.Row():
                memory_mode = gr.Dropdown(
                    choices=list(config.MEMORY_MODE_LABELS.items()),
                    value=config.MODE_BALANCED, label="Modo de memoria",
                )
                gpu_mem_limit = gr.Slider(
                    0.0, 12.0, value=0.0, step=0.5,
                    label="Límite de VRAM por sesión (GB, 0 = automático)",
                )
                force_cpu = gr.Checkbox(value=False, label="Forzar CPU (sin GPU)")
            with gr.Row():
                keep_audio = gr.Checkbox(value=True, label="Conservar audio")
                keep_fps = gr.Checkbox(value=True, label="Conservar FPS")
                output_quality = gr.Slider(
                    12, 30, value=18, step=1, label="Calidad de salida (CRF, menor = mejor)",
                )

        # ----- Lista ordenada de controles (orden = el de _build_settings) ---
        control_inputs = [
            swapper_model, enhancer_model, enhancer_blend, codeformer_fidelity,
            face_selector, reference_index, reference_distance,
            face_opacity, mask_blur, mask_padding, color_match, processing_resolution,
            temporal_smoothing, temporal_alpha,
            memory_mode, gpu_mem_limit, force_cpu,
            keep_audio, keep_fps, output_quality,
        ]

        # ----- Acciones ------------------------------------------------------
        gr.Markdown("### ▶️ Acciones")
        with gr.Row():
            n_preview = gr.Slider(2, 12, value=6, step=1, label="Frames de previsualización")
            preview_btn = gr.Button("👁️ Previsualizar frames clave", variant="secondary")
            process_btn = gr.Button("🚀 Procesar vídeo completo", variant="primary")

        status_md = gr.Markdown("")
        preview_gallery = gr.Gallery(
            label="Previsualización", columns=3, object_fit="contain"
        )
        with gr.Row():
            output_video = gr.Video(label="🎉 Resultado")
            output_file = gr.File(label="⬇️ Descargar resultado")

        # ----- Wiring --------------------------------------------------------
        refresh_btn.click(_on_refresh_system, inputs=None, outputs=system_md)

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
            "💡 **Consejo:** previsualiza primero con varios frames para afinar enhancer, "
            "máscara y opacidad antes de procesar el vídeo completo. En tarjetas de 8 GB usa "
            "el modo **Equilibrado**; si te quedas sin VRAM, baja a **Bajo VRAM** o **VRAM mínima**."
        )

    return demo
