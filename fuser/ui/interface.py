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
from .i2v_interface import build_i2v_tab

log = get_logger(__name__)

_PIPELINE_CACHE: dict = {"pipeline": None, "signature": None}


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
            "🎤 **Videos musicales** — ✅ **Se activó FaceFusion** + **post-procesado agresivo de "
            "boca/dientes** + **2 pasadas** + **RAM al máximo** + **6 referencias** recomendadas. "
            "Sube **4–6 fotos** (frontal, 3/4 y perfil; con boca abierta y cerrada) y previsualiza un "
            "frame con la boca abierta y uno de perfil."
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

        # Dos pestañas: el face swap de siempre y la función nueva
        # 'Imagen → Vídeo' (independiente: no comparte pipeline ni modelos).
        with gr.Tabs():
            with gr.Tab("🎭 Face swap (vídeo)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            source_files = gr.Files(
                                label="🙂 Imagen(es) fuente (la cara a aplicar)",
                                file_count="multiple", file_types=["image"], type="filepath",
                            )
                            # (la guía de qué fotos subir está justo debajo, en REFERENCE_TIP)
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
                recommendation_md = gr.Markdown(
                    _recommendation(config.EXPR_STANDARD, config.ENGINE_INSIGHTFACE)
                )

                # ----- Modo (caso de uso) -------------------------------------------
                with gr.Row():
                    expression_mode = gr.Dropdown(
                        choices=list(config.EXPRESSION_MODE_LABELS.items()),
                        value=config.EXPR_STANDARD,
                        label="🎚️ Modo (ajusta automáticamente la calidad para el caso de uso)",
                        info="Un clic configura motor, enhancer, máscaras, ojos/boca, 2 pasadas y RAM. "
                             "'Videos musicales' = el mejor preajuste para caras cantando.",
                    )

                # ----- Modelos y calidad --------------------------------------------
                with gr.Accordion("🎨 Modelos y calidad", open=True):
                    with gr.Row():
                        swapper_model = gr.Dropdown(
                            choices=config.SWAPPER_CHOICES, value="inswapper_128", label="Modelo de swap",
                            info="Modelo que reemplaza la identidad (motor InsightFace). "
                                 "inswapper_128 es el más compatible.",
                        )
                        enhancer_model = gr.Dropdown(
                            choices=config.ENHANCER_CHOICES, value="gfpgan_1.4",
                            label="Enhancer (restaurador de cara)",
                            info="Restaura/realza la cara tras el swap. GFPGAN = natural y rápido; "
                                 "CodeFormer = más nítido (mejor en dientes).",
                        )
                    with gr.Row():
                        enhancer_blend = gr.Slider(
                            0.0, 1.0, value=0.8, step=0.05, label="Fuerza del enhancer",
                            info="Cuánto se mezcla el realce con el swap (0 = nada, 1 = máximo).",
                        )
                        codeformer_fidelity = gr.Slider(
                            0.0, 1.0, value=0.7, step=0.05, label="Fidelidad CodeFormer (0=detalle, 1=fiel)",
                            info="Solo CodeFormer: 0 = inventa más detalle (más nítido), "
                                 "1 = más fiel al original (menos agresivo).",
                        )

                # ----- FaceFusion avanzado (opcional, colapsable) -------------------
                with gr.Accordion("⚙️ FaceFusion avanzado (resolución interna del swap)", open=False):
                    with gr.Row():
                        ff_swapper_model = gr.Dropdown(
                            choices=config.FF_SWAPPER_CHOICES, value="inswapper_128",
                            label="Modelo de swap de FaceFusion",
                            info="Modelo que usa FaceFusion para el intercambio. hyperswap = mayor resolución.",
                        )
                        ff_pixel_boost = gr.Dropdown(
                            choices=config.FF_PIXEL_BOOST_CHOICES, value="256x256",
                            label="Pixel boost (resolución interna: 256/512 = más calidad, más VRAM)",
                            info="Resolución interna a la que corre el swap. Más alto = dientes y ojos más "
                                 "nítidos, pero usa más VRAM y va más lento.",
                        )
                    with gr.Row():
                        use_mouth_pixel_boost = gr.Checkbox(
                            value=True, label="Pixel boost localizado de boca (512)",
                            info="Reprocesa la boca con CodeFormer a 512 cuando está abierta y la pega con "
                                 "máscara suave. Dientes claramente más definidos.",
                        )
                        mouth_enhancement_strength = gr.Slider(
                            0.0, 2.0, value=1.0, step=0.1, label="Fuerza del enhancer de boca",
                            info="Multiplica la fuerza del realce localizado de boca/dientes (1.0 = normal, "
                                 ">1 = más agresivo).",
                        )
                        profile_blending_strength = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05, label="Blending en perfiles laterales",
                            info="En caras de lado, recupera el borde original (mandíbula/oreja) para evitar "
                                 "deformaciones. 0 = nada, 1 = máximo.",
                        )

                # ----- Preservación de detalle --------------------------------------
                with gr.Accordion("✨ Preservación de ojos, boca y máscara", open=True):
                    with gr.Row():
                        eye_preservation = gr.Slider(
                            0.0, 1.0, value=0.4, step=0.05,
                            label="👁️ Preservación de ojos (nitidez/vida de la mirada)",
                            info="Realza los ojos para que no queden 'muertos' ni planos. Sube si la mirada "
                                 "pierde vida.",
                        )
                        mouth_detail = gr.Slider(
                            0.0, 1.0, value=0.4, step=0.05,
                            label="👄 Detalle de boca/dientes (al cantar)",
                            info="Realza dientes e interior de la boca. Actúa más fuerte cuando la boca está "
                                 "abierta (detectado por landmarks).",
                        )
                        mouth_enhancer = gr.Checkbox(
                            value=True,
                            label="🦷 Enhancer localizado de boca (CodeFormer, solo FaceFusion + boca abierta)",
                            info="2.º pase de CodeFormer aplicado SOLO en la boca cuando está abierta "
                                 "(FaceFusion). Dientes más nítidos. Desactívalo si ves artefactos.",
                        )
                    with gr.Row():
                        mask_mode = gr.Dropdown(
                            choices=list(config.MASK_MODE_LABELS.items()), value=config.MASK_HULL,
                            label="Tipo de máscara (contorno = mejor en perfiles)",
                            info="Forma de recortar la cara al pegarla. 'Contorno' sigue el rostro real "
                                 "(ideal perfiles); 'Rectángulo' es lo más básico.",
                        )
                        face_opacity = gr.Slider(
                            0.1, 1.0, value=1.0, step=0.05, label="Fuerza del swap (opacidad)",
                            info="Intensidad del intercambio. Por debajo de 1 deja ver algo de la cara original.",
                        )
                        color_match = gr.Checkbox(
                            value=False, label="Igualar color al original (iluminación)",
                            info="Adapta el tono/iluminación de la cara nueva al del vídeo. Útil con luces "
                                 "cambiantes de escenario.",
                        )
                    with gr.Row():
                        mask_blur = gr.Slider(
                            0.0, 0.8, value=0.25, step=0.05, label="Suavizado del borde (máscara)",
                            info="Difumina el borde de la máscara para que no se note la costura. "
                                 "Súbelo si ves un borde marcado.",
                        )
                        mask_padding = gr.Slider(
                            0.0, 0.4, value=0.0, step=0.02, label="Recorte interior (máscara)",
                            info="Encoge la máscara hacia dentro para no invadir pelo/frente/orejas.",
                        )

                # ----- Selección de caras y referencias -----------------------------
                with gr.Accordion("👥 Selección de caras y multi-referencia", open=False):
                    with gr.Row():
                        face_selector = gr.Dropdown(
                            choices=list(config.FACE_SELECTOR_LABELS.items()),
                            value=config.FACE_SELECTOR_ALL, label="¿A qué caras aplicar el swap?",
                            info="Todas las caras, solo la más grande, solo una persona (por referencia) "
                                 "o por posición.",
                        )
                        reference_count = gr.Dropdown(
                            choices=config.REFERENCE_COUNT_CHOICES, value=0,
                            label="Nº de imágenes de referencia a usar",
                            info="Cuántas fotos de origen combinar. Más (4–6) = más consistencia con la "
                                 "cabeza en movimiento. No cuesta VRAM extra.",
                        )
                    with gr.Row():
                        reference_index = gr.Number(
                            value=0, precision=0, label="Índice de cara (izq→der)",
                            info="Para los modos 'por posición' / 'por referencia': qué cara del vídeo, "
                                 "contando de izquierda a derecha (0 = la primera).",
                        )
                        reference_distance = gr.Slider(
                            0.5, 2.0, value=1.2, step=0.05,
                            label="Tolerancia de referencia (mayor = más permisivo)",
                            info="En modo 'por referencia': qué tan parecida debe ser una cara para "
                                 "considerarla la misma persona. Súbelo si no la detecta.",
                        )

                # ----- Estabilidad temporal -----------------------------------------
                with gr.Accordion("🎞️ Estabilidad temporal (movimiento)", open=False):
                    with gr.Row():
                        temporal_smoothing = gr.Checkbox(
                            value=True, label="Suavizado temporal",
                            info="Reduce el temblor de la cara entre frames consecutivos.",
                        )
                        temporal_alpha = gr.Slider(
                            0.0, 0.9, value=0.55, step=0.05, label="Intensidad del suavizado",
                            info="Más alto = más estable, pero puede dar 'lag' en movimientos rápidos.",
                        )
                    with gr.Row():
                        motion_adaptive = gr.Checkbox(
                            value=True, label="Adaptativo al movimiento (sin lag en la boca)",
                            info="Suaviza solo el temblor, pero reacciona al instante cuando hay movimiento "
                                 "(la boca al cantar no se arrastra).",
                        )
                        two_pass_temporal = gr.Checkbox(
                            value=False, label="2 pasadas: máxima estabilidad (usa RAM)",
                            info="Analiza un tramo entero en RAM y estabiliza los landmarks con ventana "
                                 "centrada (sin lag). Más calidad temporal; algo más lento.",
                        )

                # ----- Memoria y rendimiento ----------------------------------------
                with gr.Accordion("🧠 Memoria y rendimiento", open=False):
                    with gr.Row():
                        memory_mode = gr.Dropdown(
                            choices=list(config.MEMORY_MODE_LABELS.items()),
                            value=config.MODE_BALANCED, label="Modo de memoria (VRAM)",
                            info="Cuánta VRAM usar. Baja a 'Bajo VRAM' o 'VRAM mínima' si te da error de "
                                 "memoria de GPU.",
                        )
                        ram_mode = gr.Dropdown(
                            choices=list(config.RAM_MODE_LABELS.items()),
                            value=config.RAM_BALANCED, label="Uso de RAM (buffers / 2 pasadas)",
                            info="Cuánta RAM del sistema usar para buffers y 2 pasadas. 'Máximo' aprovecha "
                                 "32 GB+ (la GPU casi nunca espera).",
                        )
                        gpu_mem_limit = gr.Slider(
                            0.0, 12.0, value=0.0, step=0.5, label="Límite VRAM/sesión (GB, 0=auto)",
                            info="Tope de VRAM por modelo. 0 = automático según el modo de memoria.",
                        )
                    with gr.Row():
                        processing_resolution = gr.Dropdown(
                            choices=[("Nativa (máxima calidad)", 0), ("1440p", 1440), ("1080p", 1080),
                                     ("720p (rápido)", 720), ("512p (muy rápido)", 512)],
                            value=0, label="Resolución de procesamiento",
                            info="Resolución a la que se procesa y se exporta. Menor = más rápido y menos "
                                 "memoria, pero menos detalle.",
                        )
                        force_cpu = gr.Checkbox(
                            value=False, label="Forzar CPU (sin GPU)",
                            info="Procesa sin GPU. Muy lento; solo para probar la UI.",
                        )
                        keep_audio = gr.Checkbox(
                            value=True, label="Conservar audio",
                            info="Incrusta el audio original en el vídeo resultado.",
                        )
                        keep_fps = gr.Checkbox(
                            value=True, label="Conservar FPS",
                            info="Mantiene los fotogramas por segundo del vídeo original.",
                        )
                        output_quality = gr.Slider(
                            12, 30, value=18, step=1, label="Calidad de salida (CRF)",
                            info="Calidad de codificación H.264: menor = mejor calidad y archivo más pesado "
                                 "(18 es un buen punto).",
                        )
                    mem_info_md = gr.Markdown(
                        _memory_panel(config.ENGINE_INSIGHTFACE, config.RAM_BALANCED, config.MODE_BALANCED, False)
                    )

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
                ]

                # ----- Acciones ------------------------------------------------------
                gr.Markdown("### ▶️ Acciones")
                with gr.Row():
                    n_preview = gr.Slider(
                        2, 12, value=6, step=1, label="Frames de previsualización",
                        info="Cuántos frames clave del vídeo procesar para ver el resultado antes de "
                             "lanzar el vídeo completo (incluye uno con boca abierta y uno de perfil).",
                    )
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
                        engine, enhancer_model, enhancer_blend, codeformer_fidelity, mask_mode,
                        eye_preservation, mouth_detail, color_match, temporal_alpha,
                        motion_adaptive, two_pass_temporal, reference_count, ram_mode,
                        recommendation_md,
                    ],
                )
                # Al cambiar el motor: recomendación dinámica + 2 pasadas por defecto con FaceFusion.
                engine.change(
                    _on_engine_change,
                    inputs=[engine, expression_mode],
                    outputs=[recommendation_md, two_pass_temporal],
                )
                # Panel de métricas de memoria en vivo (motor / perfil de RAM / modo VRAM).
                _mem_inputs = [engine, ram_mode, memory_mode, force_cpu]
                for _comp in (engine, ram_mode, memory_mode, force_cpu):
                    _comp.change(_memory_panel, inputs=_mem_inputs, outputs=mem_info_md)

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

            with gr.Tab("🎞️ Imagen → Vídeo (Wan 2.2)"):
                build_i2v_tab()

    return demo
