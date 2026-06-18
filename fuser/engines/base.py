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

    # ----- info ---------------------------------------------------------------
    def info(self) -> EngineInfo:
        return EngineInfo(self.name, self.display_name, self.description, self.supports_two_pass())
