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

import json
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import List, Optional

import cv2
import gradio as gr
import numpy as np

from .. import __app_name__, __version__, config
from ..core import face_library
from ..core.pipeline import SwapPipeline
from ..utils import video as videoutil
from ..utils.logging import get_logger
from ..utils.system import format_system_summary

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
    s = config.Settings(
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
    # Aplica las claves AVANZADAS del preset de modo que NO tienen control en la UI
    # (ángulos de detector, umbrales, peso del enhancer). Sin esto quedaban muertas:
    # el preset solo aplicaba la mitad de sus ajustes. Solo se tocan campos SIN
    # control propio, así no pisa elecciones manuales del usuario.
    preset = config.EXPRESSION_PRESETS.get(expression_mode, {})
    for k in ("ff_detector_angles", "ff_detector_score", "ff_landmarker_score",
              "ff_temporal_fallback", "ff_enhancer_weight", "ff_geometry_mask",
              "ff_swapper_weight", "chain_shape_then_texture",
              "ff_occluder_model", "color_harmonize", "color_harmonize_strength"):
        if k in preset:
            setattr(s, k, preset[k])
    return s


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


# Valor "sin cara guardada" del desplegable de la Biblioteca de Caras.
NO_FACE = "— subir fotos —"


def _face_choices() -> list:
    """Opciones del desplegable de caras guardadas (con la opción de subir)."""
    return [NO_FACE] + face_library.list_faces()


def _resolve_source_paths(source_files, face_choice) -> List[str]:
    """Rutas de imágenes fuente: una Cara guardada si se eligió, si no lo subido."""
    if face_choice and face_choice != NO_FACE:
        paths = face_library.face_images(face_choice)
        if paths:
            return paths
        raise gr.Error(f"La cara «{face_choice}» no tiene fotos guardadas. Volvé a guardarla.")
    out = []
    for f in (source_files or []):
        p = f if isinstance(f, str) else getattr(f, "name", None)
        if p:
            out.append(p)
    return out


def _apply_dfm(settings: config.Settings, face_choice) -> Optional[str]:
    """Si la Cara elegida tiene un .dfm entrenado, activa el Deep Swapper con él.

    Debe llamarse ANTES de _get_pipeline (el model_signature incluye el .dfm, así
    que activa la recarga del pipeline con el modelo correcto). Devuelve el
    model_id del .dfm o None (one-shot normal).
    """
    dfm = face_library.dfm_of(face_choice) if (face_choice and face_choice != NO_FACE) else None
    settings.ff_deep_swapper_model = dfm or ""
    return dfm


def _prepare(pipeline: SwapPipeline, source_files, video_path, face_choice=None) -> str:
    """Valida entradas, prepara la cara fuente (multi-ref) y devuelve un resumen.

    La fuente puede venir de una **Cara guardada** (Biblioteca de Caras) o de las
    fotos subidas. ``face_choice`` tiene prioridad si apunta a una cara real.
    """
    if not video_path:
        raise gr.Error("Sube un vídeo objetivo.")
    # Deep Swapper (.dfm): la identidad vive en el modelo entrenado; NO necesita
    # fotos fuente. La Cara puede tener fotos igual (para el one-shot) pero acá se
    # ignoran; preparamos la fuente si hay, sin bloquear si falla.
    deep = getattr(pipeline.settings, "ff_deep_swapper_model", "") or ""

    source_paths = _resolve_source_paths(source_files, face_choice)
    if not source_paths and not deep:
        raise gr.Error("Elegí una Cara guardada o subí al menos una imagen fuente.")

    images = []
    for path in source_paths:
        img = _read_image(path) if path else None
        if img is not None:
            images.append(img)

    stats = None
    if images:
        try:
            stats = pipeline.prepare_source(images)
        except ValueError as exc:
            if not deep:
                raise gr.Error(str(exc))
    elif not deep:
        raise gr.Error("No se pudieron leer las imágenes fuente.")

    if deep:
        return f"🧬 Modelo entrenado (.dfm): {deep} · la geometría viene del modelo, no de fotos."

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


def _dfm_library_status() -> str:
    """Resumen de qué Caras tienen un .dfm entrenado asociado."""
    faces = face_library.list_faces()
    withdfm = [n for n in faces if face_library.has_dfm(n)]
    if not withdfm:
        return "*(ninguna Cara tiene modelo `.dfm` todavía)*"
    return "🧬 Con modelo entrenado: " + ", ".join(f"**{n}**" for n in withdfm)


def _on_import_dfm(cara, dfm_file):
    """Asocia un .dfm entrenado a una Cara existente."""
    path = dfm_file if isinstance(dfm_file, str) else getattr(dfm_file, "name", None)
    try:
        msg = face_library.set_dfm(cara, path)
    except ValueError as exc:
        return f"⚠️ {exc}", gr.update()
    return f"{msg}\n\n{_dfm_library_status()}", gr.update(value=None)


def _on_create_model(name, files, progress=gr.Progress()):
    """Pestaña 🧬 Crear modelo: cura las fotos, crea la Cara y arma el paquete
    de entrenamiento (.zip) para llevar a la nube/DeepFaceLab.

    Salidas: [cm_report, cm_bundle, cm_next, face_choice, lib_delete, dfm_cara, cm_dfm_cara].
    """
    from ..core import faceset
    name = (name or "").strip()
    nofill = (gr.update(),) * 3
    if not name:
        return ("⚠️ Poné un nombre para el modelo/persona.", None, "", *nofill)
    paths = [f if isinstance(f, str) else getattr(f, "name", None) for f in (files or [])]
    paths = [p for p in paths if p]
    if len(paths) < 10:
        return ("⚠️ Subí bastantes fotos (idealmente 500-2000; mínimo ~10 para probar el flujo).",
                None, "", *nofill)

    out_dir = config.OUTPUTS_DIR / ("faceset_" + face_library._slug(name))
    import shutil as _sh
    _sh.rmtree(out_dir, ignore_errors=True)
    progress(0.05, desc="Curando fotos…")
    rep = faceset.curate(paths, out_dir=out_dir,
                         progress=lambda f, m="": progress(0.05 + f * 0.8, desc=m))
    if rep["kept"] == 0:
        return ("No quedó ninguna foto útil:\n\n" + faceset.format_report_md(rep), None, "", *nofill)

    progress(0.9, desc="Creando la Cara…")
    try:
        face_library.save_face(name, rep["kept_paths"][:face_library.MAX_IMAGES])
    except ValueError as exc:
        return (f"⚠️ {exc}", None, "", *nofill)

    progress(0.95, desc="Armando el paquete de entrenamiento…")
    bundle = faceset.make_bundle(out_dir, name)

    report_md = ("### ✅ Curado\n\n" + faceset.format_report_md(rep)
                 + f"\n\n**Cara «{name}» creada** (con las mejores fotos, para el swap one-shot y para "
                   "colgarle el `.dfm` después).")
    next_md = (
        f"### ▶️ Siguiente\n\n"
        f"Seguí con el **paso ②** (instalar el entrenador local, una sola vez) y el **paso ③** "
        f"(preparar el entrenamiento de «{name}»). Todo desde la app.\n\n"
        f"*(El paquete de abajo es opcional: solo si algún día preferís entrenar en la nube — "
        f"ver `CLOUD_TRAIN.md`.)*"
    )
    faces = face_library.list_faces()
    return (report_md, str(bundle), next_md,
            gr.update(choices=[NO_FACE] + faces),           # face_choice
            gr.update(choices=faces),                       # lib_delete
            gr.update(choices=faces, value=name))           # cm_model_cara (pestaña 🧬)


# --- Entrenador local de .dfm (pestaña 🧬, pasos ②-⑤) ---------------------------
def _trainer_status_md() -> str:
    from ..core import dfm_trainer
    try:
        st = dfm_trainer.status()
    except Exception as exc:
        return f"⚠️ {exc}"
    build = "✅ instalado" if st["build_ready"] else "❌ falta"
    rtt = "✅ listo" if st["rtt_ready"] else "❌ falta"
    return (f"**Entrenador local** (en `{st['root']}`): build DeepFaceLab DX12 {build} · "
            f"preentrenado RTT 224 {rtt}.")


def _on_trainer_install(progress=gr.Progress()):
    from ..core import dfm_trainer
    try:
        progress(0.01, desc="Instalando el entrenador local (descarga grande, una sola vez)…")
        msg = dfm_trainer.install(progress=lambda f, m="": progress(min(f, 0.99), desc=m))
    except Exception as exc:
        log.exception("Instalación del entrenador falló")
        return f"⚠️ {exc}\n\n{_trainer_status_md()}"
    return f"{msg}\n\n{_trainer_status_md()}"


def _on_trainer_prepare(cara, dst_videos, progress=gr.Progress()):
    """Genera desde las imágenes TODO el material que DeepFaceLab necesita.

    Videos destino OPCIONALES: sin videos usa el faceset genérico (descarga
    única ~8.8 GB) y el modelo sale "universal".
    """
    from ..core import dfm_trainer
    if not cara:
        return "⚠️ Elegí la Cara (creala primero en el paso ①)."
    videos = [v if isinstance(v, str) else getattr(v, "name", None) for v in (dst_videos or [])]
    videos = [v for v in videos if v]
    src = config.OUTPUTS_DIR / ("faceset_" + face_library._slug(cara))
    if not src.is_dir():
        # sin curado previo: usa las fotos guardadas de la Cara
        src = face_library.face_dir(cara)
    try:
        return dfm_trainer.prepare(cara, src, videos,
                                   progress=lambda f, m="": progress(f, desc=m))
    except Exception as exc:
        log.exception("Preparación del entrenamiento falló")
        return f"⚠️ {exc}"


def _on_trainer_start(cara):
    from ..core import dfm_trainer
    if not cara:
        return "⚠️ Elegí la Cara."
    try:
        return (dfm_trainer.start(cara)
                + "\n\n⚠️ Mientras entrena, la GPU está ocupada: no proceses videos a la vez. "
                  "Un modelo decente lleva **días** de entrenamiento continuo (podés parar y retomar).")
    except Exception as exc:
        return f"⚠️ {exc}"


def _on_trainer_stop(cara):
    from ..core import dfm_trainer
    if not cara:
        return "⚠️ Elegí la Cara."
    try:
        return dfm_trainer.stop(cara)
    except Exception as exc:
        return f"⚠️ {exc}"


def _on_trainer_refresh(cara):
    from ..core import dfm_trainer
    if not cara:
        return "⚠️ Elegí la Cara."
    info = dfm_trainer.progress_info(cara)
    run = "🏃 ENTRENANDO" if info["running"] else "⏸️ detenido"
    lines = [f"**Estado:** {run} · fase: {info['phase']}"]
    if info["iter"] is not None:
        lines.append(f"**Iteración:** {info['iter']:,} · {info['ms']} ms/iter · "
                     f"pérdida src {info['loss_src']} / dst {info['loss_dst']}")
        lines.append("*Guía: >100k iters = parecido inicial · 300-600k = bueno · 1M+ = excelente.*")
    if info["tail"]:
        lines.append("```\n" + info["tail"][-900:] + "\n```")
    return "\n\n".join(lines)


def _on_trainer_export(cara, progress=gr.Progress()):
    from ..core import dfm_trainer
    if not cara:
        return "⚠️ Elegí la Cara.", gr.update()
    try:
        progress(0.1, desc="Exportando el modelo a .dfm (unos minutos)…")
        dfm_path = dfm_trainer.export(cara)
        progress(0.8, desc="Asociando el .dfm a la Cara…")
        msg = face_library.set_dfm(cara, str(dfm_path))
    except Exception as exc:
        log.exception("Export del .dfm falló")
        return f"⚠️ {exc}", gr.update()
    return (f"{msg}\n\n🎉 Listo: **reiniciá Fuser** y elegí «{cara}» como Cara para montar con el "
            f"modelo entrenado (geometría completa).\n\n{_dfm_library_status()}"), gr.update(value=str(dfm_path))


def _on_save_face(name, files):
    """Guarda/reemplaza una Cara de la Biblioteca y refresca los desplegables."""
    paths = [f if isinstance(f, str) else getattr(f, "name", None) for f in (files or [])]
    try:
        msg = face_library.save_face(name, paths)
    except ValueError as exc:
        return gr.update(), gr.update(), f"⚠️ {exc}", gr.update(), gr.update()
    faces = face_library.list_faces()
    return (
        gr.update(choices=[NO_FACE] + faces, value=(name or "").strip()),  # face_choice: auto-selecciona
        gr.update(choices=faces, value=None),                              # lib_delete
        msg,
        gr.update(value=None),                                             # limpia el uploader de la biblioteca
        gr.update(choices=faces),                                          # dfm_cara
    )


def _on_delete_face(name):
    """Borra una Cara guardada y refresca los desplegables.

    Salidas (orden fijo): [face_choice, lib_delete, dfm_cara, lib_status].
    """
    if not name:
        return gr.update(), gr.update(), gr.update(), "⚠️ Elegí una cara para borrar."
    try:
        msg = face_library.delete_face(name)
    except ValueError as exc:
        return gr.update(), gr.update(), gr.update(), f"⚠️ {exc}"
    faces = face_library.list_faces()
    return (
        gr.update(choices=[NO_FACE] + faces, value=NO_FACE),  # face_choice
        gr.update(choices=faces, value=None),                 # lib_delete
        gr.update(choices=faces, value=None),                 # dfm_cara
        msg,                                                  # lib_status
    )


def _apply_expression_mode(mode: str):
    """Al elegir un Modo, rellena los controles con los valores recomendados."""
    preset = config.EXPRESSION_PRESETS.get(mode, config.EXPRESSION_PRESETS[config.EXPR_STANDARD])
    eng = preset.get("engine", config.ENGINE_FACEFUSION)
    return (
        eng,
        preset.get("ff_swapper_model", "inswapper_128"),
        preset.get("ff_pixel_boost", "native"),
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
        preset.get("memory_mode", config.MODE_BALANCED),
        preset.get("reference_distance", 1.2),
        _recommendation(mode, eng),
    )


def _recommendation(mode: str, engine: str) -> str:
    """Recomendación automática según el modo y el motor elegidos (para la UI)."""
    tips = {
        config.EXPR_MAX: (
            "🔥 **MÁXIMO** — el pipeline completo de máxima fidelidad: **inswapper a "
            "pixel boost 512** (ganador medido en esta GPU) (multi-referencia con TODAS tus fotos), máscaras **xseg_2 + "
            "bisenet**, CodeFormer restaurando detalle + realce de boca/ojos + textura de "
            "piel, **armonización de color/iluminación**, 2 pasadas y **QC** anti-defectos. "
            "Es el más lento. Consejo: subí **4–8 fotos nítidas** (frontal, 3/4, perfil; "
            "boca abierta y cerrada) — sigue siendo el factor nº1 de parecido. "
            "El botón **🚀 MAXIMUM SWAP** corre esto mismo de un clic, ignorando los controles."
        ),
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
        config.EXPR_MAXIDENTITY: (
            "🎯 **Máxima Identidad** — para cuando el resultado *sigue pareciéndose a la cara del "
            "video*: **hififace** (el único modelo que **transfiere la forma de nariz/cráneo** de la "
            "foto) + máscara de geometría + **empuje anti-video del embedding** + enhancer fiel. "
            "Consejo: elegí una **Cara guardada** con varias fotos, o subí 4–8 nítidas."
        ),
        config.EXPR_MAXID_CHAIN: (
            "🎯➕ **Máxima Identidad PRO** — la cadena **forma + textura**: primero hififace impone "
            "la geometría de la foto, después inswapper re-inyecta la identidad de textura (lo mejor "
            "de ambos, medido). Es ~2× más lento que 🎯. El que más se parece a la foto."
        ),
        config.EXPR_STANDARD: (
            "**Estándar** — rápido y liviano para probar material. Para el render final usá "
            "**🎯 Máxima Identidad / PRO** (parecido a la foto) o **🔥 MÁXIMO** (nitidez tope)."
        ),
    }
    base = tips.get(mode, "")
    return f"### 💡 Recomendación\n{base}"


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


def _on_preview(source_files, face_choice, video_path, n_preview, *control_values, progress=gr.Progress()):
    settings = _build_settings(*control_values)
    _apply_dfm(settings, face_choice)
    try:
        progress(0.02, desc="Cargando modelos…")
        pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.4, desc=m))
        src = _prepare(pipeline, source_files, video_path, face_choice)

        def cb(frac, msg=""):
            progress(0.4 + frac * 0.6, desc=msg)

        results = pipeline.preview(video_path, n_frames=int(n_preview), progress=cb)
        return results, f"✅ Previsualización lista. {src}\nAjusta y vuelve a previsualizar si hace falta."
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Error en la previsualización")
        raise gr.Error(f"Error al previsualizar: {exc}")


def _on_compare(source_files, face_choice, video_path, models, n_cmp, *control_values, progress=gr.Progress()):
    """A/B: corre varios modelos de swap sobre los mismos frames del material del usuario."""
    settings = _build_settings(*control_values)
    src_paths = _resolve_source_paths(source_files, face_choice)
    if not src_paths:
        raise gr.Error("Elegí una Cara guardada o subí al menos una imagen fuente.")
    if not video_path:
        raise gr.Error("Subí un video objetivo.")
    models = [m for m in (models or [])]
    if len(models) < 2:
        raise gr.Error("Elegí al menos 2 modelos para comparar.")

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


def _release_pipeline() -> None:
    """Descarga los modelos del pipeline cacheado y limpia el pool de FaceFusion.

    Imprescindible entre pasadas de la cadena: sin el clear del inference-pool de
    FF (que hace ``engine.unload``), la 2ª pasada reusaría el pool del modelo de la
    1ª (bug del pool obsoleto). También libera VRAM antes de cargar el 2º modelo.
    """
    p = _PIPELINE_CACHE.get("pipeline")
    if p is not None and getattr(p, "engine", None) is not None:
        try:
            p.engine.unload()
        except Exception:  # pragma: no cover
            pass
    _PIPELINE_CACHE["pipeline"] = None
    _PIPELINE_CACHE["signature"] = None


def _swap_video_once(settings, source_files, face_choice, video_path, progress, lo, hi, tag):
    """Una pasada completa (cargar modelos → preparar fuente → procesar vídeo)."""
    span = hi - lo
    pipeline = _get_pipeline(
        settings, progress=lambda f, m="": progress(lo + span * f * 0.15, desc=f"{tag} · {m}"))
    src = _prepare(pipeline, source_files, video_path, face_choice)
    out = pipeline.process_video(
        video_path, progress=lambda f, m="": progress(lo + span * (0.15 + 0.85 * f), desc=f"{tag} · {m}"))
    return out, src


def _process_chain(base, source_files, face_choice, video_path, progress):
    """Cadena forma→textura (Máxima Identidad PRO): hififace y luego inswapper.

    Pasada 1 (hififace + máscara BOX, SIN enhancer/temporal/QC) impone la forma de
    nariz/cráneo de la foto. Pasada 2 (inswapper + enhancer + temporal/QC) reinyecta
    textura/nitidez tratando la salida de la pasada 1 como objetivo → preserva la
    forma nueva. Se libera el modelo entre pasadas (VRAM + pool de FF).
    """
    pass1 = replace(
        base, ff_swapper_model="hififace_unofficial_256", ff_pixel_boost="512x512",
        ff_geometry_mask=True, enhancer_model="none", enhancer_blend=0.0,
        two_pass_temporal=False, qc_second_pass=False, temporal_smoothing=False,
        chain_shape_then_texture=False,
    )
    out1, _ = _swap_video_once(pass1, source_files, face_choice, video_path, progress,
                               0.0, 0.5, "🎯➕ Pasada 1/2 · forma (hififace)")
    _release_pipeline()  # limpia pool de FF + VRAM antes del 2º modelo
    pass2 = replace(base, ff_swapper_model="inswapper_128", chain_shape_then_texture=False)
    out2, src = _swap_video_once(pass2, source_files, face_choice, out1, progress,
                                 0.5, 1.0, "🎯➕ Pasada 2/2 · textura (inswapper)")
    return out2, src


def _on_process(source_files, face_choice, video_path, *control_values, progress=gr.Progress()):
    settings = _build_settings(*control_values)
    _apply_dfm(settings, face_choice)
    try:
        if getattr(settings, "chain_shape_then_texture", False):
            progress(0.0, desc="🎯➕ Cadena forma+textura (2 pasadas)…")
            out_path, src = _process_chain(settings, source_files, face_choice, video_path, progress)
            return out_path, out_path, f"✅ ¡Vídeo procesado! (cadena forma+textura) {src}  ·  Descárgalo abajo."

        progress(0.0, desc="Cargando modelos…")
        pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.15, desc=m))
        src = _prepare(pipeline, source_files, video_path, face_choice)

        def cb(frac, msg=""):
            progress(0.15 + frac * 0.85, desc=msg)

        out_path = pipeline.process_video(video_path, progress=cb)
        return out_path, out_path, f"✅ ¡Vídeo procesado! {src}  ·  Descárgalo abajo."
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Error al procesar el vídeo")
        raise gr.Error(f"Error al procesar: {exc}")


def _max_settings() -> config.Settings:
    """Settings del 🚀 MAXIMUM SWAP: la opción nuclear, IGNORA los controles.

    Parte de los defaults, aplica el preset EXPR_MAX completo (todas sus claves
    mapean 1:1 a ``Settings``) y remata con resolución/calidad de salida al tope
    y la arena de VRAM del modo de máxima calidad. Reproducible: el handler
    guarda el dict exacto usado en ``tmp/maximum_swap_*.json``.
    """
    s = config.Settings()
    s.expression_mode = config.EXPR_MAX
    for key, value in config.EXPRESSION_PRESETS[config.EXPR_MAX].items():
        setattr(s, key, value)
    s.processing_resolution = 1080     # máxima resolución práctica de trabajo
    # ¡OJO! output_quality es el CRF de x264 (MENOR = mejor). 10 ≈ casi sin
    # pérdida. (Poner 98 aquí fue el bug del "video deformado": x264 lo clampa
    # a 51 = peor calidad posible y el encoder tritura TODO el frame.)
    s.output_quality = 10
    s.gpu_mem_limit_gb = config.MEMORY_PRESETS[config.MODE_MAX_QUALITY]["gpu_mem_limit_gb"]
    return s


def _on_maximum_swap(source_files, face_choice, video_path, progress=gr.Progress()):
    """🚀 MAXIMUM SWAP: pipeline completo de máxima fidelidad, de un clic.

    Etapas: (1) carga de modelos → (2) swap multi-referencia con pixel boost 512
    + máscaras xseg_2/bisenet → (3) restauración CodeFormer + realce localizado
    de boca/ojos + textura de piel → (4) armonización fotométrica LAB → (5) 2ª
    pasada temporal + QC de defectos. Devuelve además un ANTES/DESPUÉS del frame
    central y registra los ajustes exactos (reproducibilidad).
    """
    settings = _max_settings()
    _apply_dfm(settings, face_choice)
    try:
        progress(0.0, desc="🚀 Etapa 1/5 · Cargando modelos (swap + xseg_2 + bisenet + CodeFormer)…")
        pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.12, desc=f"🚀 Etapa 1/5 · {m}"))
        src = _prepare(pipeline, source_files, video_path, face_choice)

        def cb(frac, msg=""):
            progress(0.12 + frac * 0.88, desc=f"🚀 {msg}")

        out_path = pipeline.process_video(video_path, progress=cb)

        # Antes/Después del frame central para el slider comparador.
        before_after = None
        try:
            info = videoutil.probe(video_path)
            mid = max(0, info.frame_count // 2)
            b = videoutil.get_frames_at(video_path, [mid])
            a = videoutil.get_frames_at(out_path, [mid])
            if b and a:
                h = min(b[0].shape[0], a[0].shape[0])
                bb = cv2.resize(b[0], (max(2, int(b[0].shape[1] * h / b[0].shape[0])), h))
                aa = cv2.resize(a[0], (bb.shape[1], h))
                before_after = (cv2.cvtColor(bb, cv2.COLOR_BGR2RGB),
                                cv2.cvtColor(aa, cv2.COLOR_BGR2RGB))
        except Exception as exc:  # pragma: no cover - solo cosmético
            log.warning("No pude armar el antes/después: %s", exc)

        # Reproducibilidad: settings EXACTOS de esta pasada.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_name = f"maximum_swap_{stamp}.json"
        try:
            config.ensure_dirs()
            (config.TEMP_DIR / log_name).write_text(
                json.dumps(asdict(settings), indent=2, default=str), encoding="utf-8")
        except Exception:  # pragma: no cover
            log_name = "(no se pudo guardar)"

        status = (
            f"🚀 ✅ **MAXIMUM SWAP** terminado. {src}\n\n"
            f"Pipeline: **{settings.ff_swapper_model} @ {settings.ff_pixel_boost}** · "
            f"máscaras **xseg_2 + bisenet_34** · CodeFormer (peso {settings.ff_enhancer_weight}) "
            f"+ realce de boca/ojos · **armonización LAB** · 2 pasadas + **QC**.  "
            f"Ajustes exactos: `tmp/{log_name}`"
        )
        return out_path, out_path, status, before_after
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Error en MAXIMUM SWAP")
        raise gr.Error(f"Error en MAXIMUM SWAP: {exc}")


def _queue_duration_seconds(video_path: str) -> float:
    """Duración del video en segundos (para ordenar la cola). Los ilegibles van al final."""
    try:
        info = videoutil.probe(video_path)
        return info.frame_count / (info.fps or 25.0)
    except Exception:
        return float("inf")


def _on_process_queue(source_files, face_choice, video_queue, *control_values, progress=gr.Progress()):
    """Procesa una COLA de videos con la MISMA cara fuente (subida o de Biblioteca).

    - Los modelos se cargan una sola vez y se reutilizan (pipeline cacheado).
    - La cola se ordena del video MÁS CORTO al MÁS LARGO antes de empezar.
    - Cada resultado aparece para descargar EN CUANTO termina (no al final).
    - Si un video falla, se manda al final y se reintenta (hasta QUEUE_MAX_ATTEMPTS).

    Es un *generator*: va emitiendo (resultados, estado) tras cada video.
    """
    settings = _build_settings(*control_values)
    _apply_dfm(settings, face_choice)
    source_paths = _resolve_source_paths(source_files, face_choice)
    if not source_paths:
        raise gr.Error("Elegí una Cara guardada o subí al menos una imagen fuente.")
    videos = [v if isinstance(v, str) else getattr(v, "name", None) for v in (video_queue or [])]
    videos = [v for v in videos if v]
    if not videos:
        raise gr.Error("Agrega al menos un video a la cola.")

    progress(0.0, desc="Cargando modelos…")
    pipeline = _get_pipeline(settings, progress=lambda f, m="": progress(f * 0.06, desc=m))

    images = []
    for path in source_paths:
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
/* 🚀 MAXIMUM SWAP: el botón "nuclear" — inconfundible */
#max-swap-btn {
  background: linear-gradient(135deg, #ff7a18 0%, #ff2d55 60%, #c81d8f 100%) !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 800 !important;
  font-size: 1.06rem !important;
  padding: 14px 18px !important;
  letter-spacing: 0.3px;
  box-shadow: 0 6px 22px rgba(255, 45, 85, 0.35) !important;
}
#max-swap-btn:hover {
  filter: brightness(1.08);
  box-shadow: 0 10px 28px rgba(255, 45, 85, 0.5) !important;
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
                    face_choice = gr.Dropdown(
                        choices=_face_choices(), value=NO_FACE,
                        label="🗂️ Cara guardada (Biblioteca)",
                        info="Elegí una cara ya guardada y te salteás subir fotos. "
                             "Si elegís una, tiene prioridad sobre las imágenes de abajo.",
                    )
                    source_files = gr.Files(
                        label="…o subí imagen(es) fuente (la cara a aplicar)",
                        file_count="multiple", file_types=["image"], type="filepath",
                    )
                    gr.Markdown(REFERENCE_TIP, elem_classes="fuser-soft")
                with gr.Accordion("🗂️ Biblioteca de Caras — crear / borrar", open=False):
                    gr.Markdown(
                        "Guardá una persona con **varias fotos** (distintos ángulos y "
                        "expresiones = más identidad). Después la elegís arriba y solo subís el video. "
                        "*(No entrena un modelo `.dfm`; guarda la identidad multi-referencia.)*",
                        elem_classes="fuser-soft",
                    )
                    lib_name = gr.Textbox(label="Nombre de la cara", placeholder="Cara 1")
                    lib_files = gr.Files(
                        label="Fotos de esta persona (varias)",
                        file_count="multiple", file_types=["image"], type="filepath",
                    )
                    lib_save_btn = gr.Button("💾 Guardar cara", variant="secondary")
                    with gr.Row():
                        lib_delete = gr.Dropdown(
                            choices=face_library.list_faces(), label="Borrar cara guardada", scale=3,
                        )
                        lib_delete_btn = gr.Button("🗑️ Borrar", scale=1)
                    lib_status = gr.Markdown("", elem_classes="fuser-soft")
                    gr.Markdown(
                        "🧬 ¿Querés un **modelo entrenado** de una persona (máxima geometría)? "
                        "Pestaña **«Crear modelo (.dfm)»** abajo.", elem_classes="fuser-soft",
                    )
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
                        _recommendation(config.EXPR_STANDARD, config.ENGINE_FACEFUSION),
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
                    ba_slider = gr.ImageSlider(
                        label="🆚 Antes / Después (frame central — arrastrá el divisor)",
                        type="numpy", visible=True,
                    )
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
                    # FaceFusion es el ÚNICO motor. engine/swapper_model quedan como
                    # ESTADO OCULTO (no se eligen en la UI) para no duplicar el selector
                    # de modelo: el swap se elige UNA sola vez con "Modelo de swap".
                    engine = gr.State(config.ENGINE_FACEFUSION)
                    swapper_model = gr.State("inswapper_128")
                    with gr.Row():
                        ff_swapper_model = gr.Dropdown(
                            choices=config.FF_SWAPPER_CHOICES, value="inswapper_128",
                            label="🧬 Modelo de swap",
                            info="El modelo que reemplaza la identidad. inswapper_128 = el más ESTABLE "
                                 "(no se 'mueve'). hififace / ghost_2 / ghost_3 (256 px) = más "
                                 "identidad y forma (mejor con Pixel boost 512). El preset 🔥 MÁXIMO "
                                 "usa hififace + 512.",
                        )
                        ff_pixel_boost = gr.Dropdown(
                            choices=config.FF_PIXEL_BOOST_CHOICES, value="native",
                            label="🔎 Pixel boost (resolución interna del swap)",
                            info="Auto = nativa del modelo (rápido). 512 = más detalle fino (la cara "
                                 "se procesa en 4 pasadas entrelazadas → swap más lento). Combo de "
                                 "máxima calidad: hififace + 512 + máscara bisenet.",
                        )
                    with gr.Row():
                        enhancer_model = gr.Dropdown(
                            choices=config.ENHANCER_CHOICES, value="gfpgan_1.4",
                            label="Enhancer (restaurador)",
                            info="GFPGAN = natural y rápido; CodeFormer = más nítido (mejor en dientes).",
                        )
                        enhancer_blend = gr.Slider(
                            0.0, 1.0, value=0.8, step=0.05, label="Fuerza del enhancer",
                            info="Cuánto se mezcla el realce (0 = nada, 1 = máximo).",
                        )
                    with gr.Row():
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

                with gr.Accordion("⚙️ FaceFusion avanzado (boca / perfiles)", open=False):
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
                        _memory_panel(config.ENGINE_FACEFUSION, config.RAM_BALANCED,
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
                    process_btn = gr.Button("Procesar vídeo completo", variant="primary")
                    max_swap_btn = gr.Button(
                        "🚀 MAXIMUM SWAP — máxima fidelidad (lento)",
                        variant="primary", elem_id="max-swap-btn",
                    )
                    gr.Markdown(
                        "_**MAXIMUM SWAP** ignora los controles y corre el pipeline completo de "
                        "máxima calidad: swap a **pixel boost 512** multi-referencia, máscaras "
                        "**xseg_2 + bisenet**, CodeFormer + realce de boca/ojos + textura de piel, "
                        "**armonización de color/iluminación**, 2 pasadas temporales y **QC** "
                        "anti-defectos. Tarda varias veces más que el modo normal._",
                        elem_classes="fuser-soft",
                    )

        # ===== Más modos: comparar modelos · herramientas =====
        gr.Markdown("### 🧭 Más modos y herramientas")
        with gr.Tabs():
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
            with gr.Tab("🧬 Crear modelo (.dfm)"):
                gr.Markdown(
                    "Creá un **modelo `.dfm`** de una persona **todo desde la app**: cargás muchas fotos, "
                    "la app las cura, **entrena en TU GPU** (DeepFaceLab DirectX12, en segundo plano) y al "
                    "final exporta e importa el modelo. La Cara resultante monta con **geometría de cráneo "
                    "completa** en cualquier video. ⏱️ Realidad: el entrenamiento lleva **días** de GPU "
                    "(podés parar/retomar y usar la PC; evitá procesar videos mientras entrena)."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### ① Cargar fotos → curar → crear la Cara")
                        cm_name = gr.Textbox(label="Nombre de la persona/modelo", placeholder="Cara 1")
                        cm_files = gr.Files(
                            label="Muchas fotos de la persona (500-2000 ideal; ángulos/expresiones variados)",
                            file_count="multiple", file_types=["image"], type="filepath",
                        )
                        cm_build_btn = gr.Button("① Curar fotos + crear Cara", variant="primary")
                        cm_report = gr.Markdown("", elem_classes="fuser-soft")
                        with gr.Accordion("Paquete para entrenar en la nube (opcional)", open=False):
                            cm_bundle = gr.File(label="⬇️ Paquete de fotos curadas (ver CLOUD_TRAIN.md)")
                        cm_next = gr.Markdown("", elem_classes="fuser-soft")

                        gr.Markdown("#### ② Instalar el entrenador local (una sola vez)")
                        gr.Markdown(
                            "Descarga automática del build **DeepFaceLab DirectX12** + el preentrenado "
                            "**RTT 224** (~4 GB en total, a `E:\\modelos\\deepfacelab`). Sin pasos manuales.",
                            elem_classes="fuser-soft",
                        )
                        cm_install_btn = gr.Button("② Instalar entrenador local", variant="secondary")
                        cm_install_status = gr.Markdown(_trainer_status_md(), elem_classes="fuser-soft")
                    with gr.Column():
                        gr.Markdown("#### ③ Preparar → ④ Entrenar → ⑤ Exportar")
                        cm_model_cara = gr.Dropdown(
                            choices=face_library.list_faces(), label="Cara / modelo a entrenar")
                        cm_dst_videos = gr.Files(
                            label="Videos destino (OPCIONAL): si los cargás, el modelo aprende las "
                                  "condiciones de ESOS videos. Si lo dejás vacío, la app usa un set "
                                  "genérico (descarga única ~8.8 GB) y el modelo sirve para CUALQUIER video.",
                            file_count="multiple", file_types=["video"], type="filepath",
                        )
                        cm_prepare_btn = gr.Button("③ Preparar entrenamiento", variant="secondary")
                        with gr.Row():
                            cm_train_btn = gr.Button("④ ▶️ Entrenar", variant="primary")
                            cm_stop_btn = gr.Button("⏸️ Parar", variant="secondary")
                            cm_refresh_btn = gr.Button("🔄 Estado", variant="secondary")
                        cm_train_status = gr.Markdown("", elem_classes="fuser-soft")
                        cm_export_btn = gr.Button("⑤ 📦 Exportar .dfm e importarlo a la Cara",
                                                  variant="primary")
                        cm_import_status = gr.Markdown(_dfm_library_status(), elem_classes="fuser-soft")
                        with gr.Accordion("Importar un .dfm entrenado afuera (opcional)", open=False):
                            cm_dfm_file = gr.File(label=".dfm", file_types=[".dfm"], type="filepath")
                            cm_import_btn = gr.Button("Importar este .dfm a la Cara elegida")

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
                memory_mode, reference_distance,
                recommendation_md,
            ],
        )
        # (Sin selector de motor: FaceFusion es el único; ``engine`` es State oculto.)
        _mem_inputs = [engine, ram_mode, memory_mode, force_cpu]
        for _comp in (ram_mode, memory_mode, force_cpu):
            _comp.change(_memory_panel, inputs=_mem_inputs, outputs=mem_info_md)

        # Biblioteca de Caras: guardar / borrar refrescan todos los desplegables.
        lib_save_btn.click(
            _on_save_face,
            inputs=[lib_name, lib_files],
            outputs=[face_choice, lib_delete, lib_status, lib_files, cm_model_cara],
        )
        lib_delete_btn.click(
            _on_delete_face,
            inputs=lib_delete,
            outputs=[face_choice, lib_delete, cm_model_cara, lib_status],
        )
        # Pestaña "🧬 Crear modelo (.dfm)"
        cm_build_btn.click(
            _on_create_model,
            inputs=[cm_name, cm_files],
            outputs=[cm_report, cm_bundle, cm_next, face_choice, lib_delete, cm_model_cara],
        )
        cm_install_btn.click(_on_trainer_install, inputs=None, outputs=cm_install_status)
        cm_prepare_btn.click(
            _on_trainer_prepare,
            inputs=[cm_model_cara, cm_dst_videos],
            outputs=cm_train_status,
        )
        cm_train_btn.click(_on_trainer_start, inputs=cm_model_cara, outputs=cm_train_status)
        cm_stop_btn.click(_on_trainer_stop, inputs=cm_model_cara, outputs=cm_train_status)
        cm_refresh_btn.click(_on_trainer_refresh, inputs=cm_model_cara, outputs=cm_train_status)
        cm_export_btn.click(
            _on_trainer_export,
            inputs=cm_model_cara,
            outputs=[cm_import_status, cm_dfm_file],
        )
        cm_import_btn.click(
            _on_import_dfm,
            inputs=[cm_model_cara, cm_dfm_file],
            outputs=[cm_import_status, cm_dfm_file],
        )

        preview_btn.click(
            _on_preview,
            inputs=[source_files, face_choice, target_video, n_preview, *control_inputs],
            outputs=[preview_gallery, status_md],
        )
        cmp_btn.click(
            _on_compare,
            inputs=[source_files, face_choice, target_video, cmp_models, cmp_frames, *control_inputs],
            outputs=[cmp_gallery, cmp_status],
        )
        process_btn.click(
            _on_process,
            inputs=[source_files, face_choice, target_video, *control_inputs],
            outputs=[output_video, output_file, status_md],
        )
        max_swap_btn.click(
            _on_maximum_swap,
            inputs=[source_files, face_choice, target_video],   # nuclear: ignora los controles
            outputs=[output_video, output_file, status_md, ba_slider],
        )
        queue_btn.click(
            _on_process_queue,
            inputs=[source_files, face_choice, video_queue, *control_inputs],
            outputs=[queue_results, queue_status],
        )
        split_btn.click(
            _on_split_video,
            inputs=[split_video],
            outputs=[split_results, split_status],
        )

    return demo
