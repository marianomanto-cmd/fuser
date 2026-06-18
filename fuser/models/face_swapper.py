"""Face swapper basado en InsightFace InSwapper (inswapper_128).

Usamos ``paste_back=False`` para obtener la cara generada **alineada** (128x128)
junto con la matriz afín. Así controlamos nosotros el pegado final (máscara,
opacidad, realce a 512 px), en lugar de delegar en el pegado básico de la
librería. Esto es clave para la calidad.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


class FaceSwapper:
    """Envoltorio de ``insightface.model_zoo`` InSwapper."""

    def __init__(self, model_path: str, providers: list):
        self._path = str(model_path)
        self._providers = providers
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from insightface import model_zoo  # import perezoso

        log.info("Cargando swapper: %s", self._path)
        self._model = model_zoo.get_model(self._path, providers=self._providers)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def input_size(self) -> int:
        return int(self._model.input_size[0]) if self._model else 128

    def swap_raw(self, frame: np.ndarray, target_face, source_face) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve ``(cara_generada_bgr, matriz_afin)`` sin pegar al frame.

        - ``cara_generada_bgr``: recorte alineado a ``input_size`` (128) en BGR.
        - ``matriz_afin``: 2x3 que mapea frame -> recorte (se invierte al pegar).
        """
        if self._model is None:
            self.load()
        bgr_fake, affine = self._model.get(frame, target_face, source_face, paste_back=False)
        return bgr_fake, affine

    def unload(self) -> None:
        self._model = None
