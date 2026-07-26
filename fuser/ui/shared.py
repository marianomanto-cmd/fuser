"""Plumbing compartido entre las pestañas de la UI (interface.py y montador.py).

Existe para que el Montador (pestaña propia, módulo propio) y el resto de la UI
compartan UNA sola caché de pipeline (dos cachés = dos juegos de modelos en
VRAM) y las mismas guardas, sin imports circulares: este módulo no importa a
ninguno de los dos.
"""
from __future__ import annotations

import gradio as gr

from .. import config
from ..core import face_library
from ..core.pipeline import SwapPipeline
from ..utils.logging import get_logger

log = get_logger(__name__)

# Caché ÚNICA del pipeline de swap: solo se recargan modelos si cambia la firma
# (modelo de swap, .dfm, memoria...); el resto de ajustes se aplica en caliente.
PIPELINE_CACHE: dict = {"pipeline": None, "signature": None}


def get_pipeline(settings: config.Settings, progress=None) -> SwapPipeline:
    pipeline = SwapPipeline(settings)
    signature = pipeline.model_signature()
    cached = PIPELINE_CACHE["pipeline"]
    if cached is not None and PIPELINE_CACHE["signature"] == signature:
        cached.update_runtime(settings)
        return cached
    if cached is not None:
        # Descargar el pipeline viejo ANTES de cargar el nuevo: los pools de
        # FaceFusion son a nivel de módulo y sin unload() quedaban vivos (VRAM
        # retenida + sesiones ONNX duplicadas al cambiar de modelo/preset).
        try:
            if getattr(cached, "engine", None) is not None:
                cached.engine.unload()
        except Exception as exc:  # pragma: no cover
            log.warning("No pude descargar el pipeline anterior: %s", exc)
        PIPELINE_CACHE["pipeline"] = None
        PIPELINE_CACHE["signature"] = None
    pipeline.load_models(progress=progress)
    PIPELINE_CACHE["pipeline"] = pipeline
    PIPELINE_CACHE["signature"] = signature
    return pipeline


# Valor "sin modelo entrenado" de los desplegables de modelos .dfm.
NO_DFM = "— sin modelo (one-shot con fotos) —"


def dfm_choices() -> list:
    """Modelos .dfm disponibles (Caras con .dfm asociado + custom/ sueltos)."""
    out = [NO_DFM]
    seen = set()
    for n in face_library.list_faces():
        mid = face_library.dfm_of(n)
        if mid:
            out.append((f"🧬 {n}", mid))
            seen.add(mid)
    try:
        for f in sorted(face_library.FF_CUSTOM_DFM_DIR.glob("*.dfm")):
            mid = f"custom/{f.stem}"
            if mid not in seen:
                out.append((f"🧬 {f.stem}", mid))
    except Exception:
        pass
    return out


def fmt_eta(secs: float) -> str:
    secs = int(max(0, secs))
    h, m = secs // 3600, (secs % 3600) // 60
    return f"{h} h {m:02d} min" if h else f"{m} min"


def bg_status_md() -> str:
    """🚦 Semáforo de procesos de fondo, siempre visible en la cabecera.

    Muestra si hay un entrenamiento DFL vivo (son subprocesos desacoplados:
    sobreviven a cierres/reinicios de la app, así que es fácil olvidar que
    existen — y con la GPU tomada, montar tira abajo la app). Lo refresca un
    gr.Timer; una sola consulta barrida a Windows por tick (_live_train_pids).
    """
    from ..core import dfm_trainer
    try:
        vivos = dfm_trainer.running_trainings()
    except Exception as exc:  # pragma: no cover
        return f"⚠️ No pude consultar los procesos de fondo: {exc}"
    if vivos:
        nombres = ", ".join(f"«{n}»" for n in vivos)
        return (f"🔴 **Fondo: ENTRENANDO a {nombres}** — la GPU está tomada "
                f"(~7,6/8,2 GB). ⏸️ Pausá antes de montar o la app se cae.")
    return "🟢 **Fondo: GPU libre** — sin entrenamientos corriendo; podés montar."


def ensure_gpu_libre(accion: str = "el montaje") -> None:
    """Bloquea las acciones de GPU (montaje/preview/cola) si hay un entrenamiento vivo.

    Con el optimizador en GPU, un entrenamiento DFL ocupa ~7,6 de los 8,2 GB de
    VRAM. Si en ese estado se intenta un swap, DirectML no puede reservar memoria
    al cargar los modelos y MATA EL PROCESO ENTERO sin excepción de Python — el
    usuario ve "Broken Connection" y la app muerta (pasó 2026-07-26, dos veces,
    con el crash capturado justo en la carga del detector). Mejor un mensaje
    claro ANTES de tocar la GPU.
    """
    from ..core import dfm_trainer
    try:
        vivos = dfm_trainer.running_trainings()
    except Exception:
        vivos = []
    if vivos:
        raise gr.Error(
            f"La GPU está entrenando a «{vivos[0]}» (usa casi toda la VRAM) y "
            f"lanzar {accion} en ese estado tira abajo la app entera. "
            f"Pausá el entrenamiento (⏸️ en su pestaña), hacé {accion}, y "
            f"retomalo después: el autoguardado cada 5 min hace que pierdas "
            f"minutos, no horas.")
