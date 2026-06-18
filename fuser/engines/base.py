"""Interfaz abstracta de un motor de face swap.

Cualquier motor (InsightFace, FaceFusion, …) implementa esta interfaz. El
``pipeline`` la usa para:

- cargar modelos (``load``),
- preparar la(s) cara(s) fuente (``prepare_source``),
- procesar un frame completo (``process_frame``).

Para el modo de **2 pasadas** (suavizado temporal centrado que usa RAM) el motor
puede, opcionalmente, exponer ``detect`` / ``select_targets`` / ``render`` y
declarar ``supports_two_pass() -> True``. Si no, el pipeline cae a 1 pasada.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class EngineInfo:
    name: str
    display_name: str
    description: str
    supports_two_pass: bool


class BaseFaceSwapper(ABC):
    #: identificador interno (config.ENGINE_*)
    name: str = "base"
    display_name: str = "Base"
    description: str = ""

    def __init__(self, settings, memory_manager):
        self.settings = settings
        self.mm = memory_manager

    # ----- ciclo de vida -------------------------------------------------------
    @abstractmethod
    def load(self, progress=None) -> None:
        """Descarga/instancia los modelos del motor."""

    @property
    @abstractmethod
    def loaded(self) -> bool:
        ...

    def update_runtime(self, settings) -> None:
        """Aplica ajustes ligeros sin recargar modelos."""
        self.settings = settings

    def unload(self) -> None:
        ...

    # ----- fuente / referencia -------------------------------------------------
    @abstractmethod
    def prepare_source(self, images: List[np.ndarray]):
        """Prepara la identidad fuente (multi-referencia). Devuelve stats o None."""

    def set_reference(self, frame: np.ndarray, index: int = 0) -> bool:
        return False

    @property
    def has_source(self) -> bool:
        """True si ya hay una identidad fuente preparada."""
        return False

    # ----- procesamiento -------------------------------------------------------
    @abstractmethod
    def process_frame(self, frame: np.ndarray, use_smoothing: bool = True) -> np.ndarray:
        """Procesa un frame completo y devuelve el frame con el swap aplicado."""

    # ----- hooks opcionales para 2 pasadas ------------------------------------
    def supports_two_pass(self) -> bool:
        return False

    def detect(self, frame: np.ndarray) -> List:
        raise NotImplementedError

    def select_targets(self, faces: List) -> List:
        raise NotImplementedError

    def render(self, frame: np.ndarray, targets: List) -> np.ndarray:
        raise NotImplementedError

    def reset_temporal(self) -> None:
        ...

    # ----- capacidades y recursos (contratos explícitos) ----------------------
    def supports_adaptive_mouth(self) -> bool:
        """True si el motor puede realzar la boca/dientes localmente (boca abierta)."""
        return False

    def enhance_mouth_region(self, frame: np.ndarray, face, openness: float = 1.0) -> np.ndarray:
        """Realce localizado de boca/dientes para una cara (opcional). Por defecto, no-op."""
        return frame

    def supports_region_enhancement(self) -> bool:
        """True si el motor puede realzar regiones concretas (boca/ojos) por separado."""
        return self.supports_adaptive_mouth()

    def prefers_two_pass(self) -> bool:
        """True si el motor se beneficia de 2 pasadas por defecto (más calidad base)."""
        return False

    def get_capabilities(self) -> dict:
        """Capacidades del motor para que el pipeline decida el flujo."""
        return {
            "adaptive_mouth": self.supports_adaptive_mouth(),
            "region_enhancement": self.supports_region_enhancement(),
            "multi_pass_temporal": self.supports_two_pass(),
            "prefers_two_pass": self.prefers_two_pass(),
            "high_res_region": False,
        }

    def get_memory_usage(self) -> dict:
        """Reporta consumo aproximado de memoria del motor (para métricas en la UI)."""
        return {"engine": self.name, "loaded": self.loaded}

    def get_memory_profile(self) -> dict:
        """Alias semántico de ``get_memory_usage`` (contrato de la interfaz)."""
        return self.get_memory_usage()

    def cleanup(self) -> None:
        """Libera los recursos del motor (modelos/sesiones)."""
        self.unload()

    # ----- info ---------------------------------------------------------------
    def info(self) -> EngineInfo:
        return EngineInfo(self.name, self.display_name, self.description, self.supports_two_pass())
