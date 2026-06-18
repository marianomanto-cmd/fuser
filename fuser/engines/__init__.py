"""Motores de face swap intercambiables.

Expone una fábrica ``create_engine`` y la interfaz ``BaseFaceSwapper``. El
``pipeline`` habla siempre con la interfaz, nunca con una implementación
concreta, de modo que cambiar de motor (InsightFace ↔ FaceFusion) es trivial.
"""
from __future__ import annotations

from .base import BaseFaceSwapper, EngineInfo
from .. import config


ENGINE_REGISTRY = {
    config.ENGINE_INSIGHTFACE: ("InsightFace (Rápido)",
                                "Rápido y ligero (~menos VRAM). Buen resultado general."),
    config.ENGINE_FACEFUSION: ("FaceFusion (Alta Calidad)",
                               "Mejor en boca abierta, dientes y perfiles. Más lento y usa más VRAM."),
}


def create_engine(settings: config.Settings, memory_manager) -> BaseFaceSwapper:
    """Instancia el motor indicado por ``settings.engine``."""
    name = settings.engine
    if name == config.ENGINE_FACEFUSION:
        from .facefusion_engine import FaceFusionSwapper
        return FaceFusionSwapper(settings, memory_manager)
    # Por defecto / fallback: InsightFace.
    from .insightface_engine import InsightFaceSwapper
    return InsightFaceSwapper(settings, memory_manager)


__all__ = ["BaseFaceSwapper", "EngineInfo", "create_engine", "ENGINE_REGISTRY"]
