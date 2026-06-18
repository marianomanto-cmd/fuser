"""Segmentación facial (BiSeNet) como ONNX — OPCIONAL.

Si el modelo está disponible, produce máscaras por región de máxima precisión
(piel, ojos, cejas, nariz, boca, labios, orejas, pelo…), lo que permite:

- Pegar el swap **solo sobre la piel real** (excelente en perfiles y con pelo
  cruzando la cara → sin invadir oreja/pelo/fondo).
- Aislar **ojos** y **boca** para realce/preservación dirigidos.

Convención CelebAMask-HQ (19 clases). El pipeline degrada con elegancia a la
máscara por contorno (casco de landmarks) si este modelo no está o falla.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from ..config import ModelInfo
from ..utils.logging import get_logger

log = get_logger(__name__)

# Índices de clase (CelebAMask-HQ / BiSeNet)
_SKIN = 1
_BROWS = (2, 3)
_EYES = (4, 5, 6)        # ojo izq, ojo der, gafas
_NOSE = (10,)
_MOUTH = (11, 12, 13)    # boca, labio superior, labio inferior
_EARS = (7, 8, 9)

# Regiones que forman la "cara" a reemplazar (sin pelo/cuello/ropa/sombrero).
_FACE_CLASSES = (_SKIN,) + _BROWS + _EYES + _NOSE + _MOUTH

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceParser:
    def __init__(self, model_path: str, providers: list, info: ModelInfo, sess_options=None):
        self._path = str(model_path)
        self._providers = providers
        self._info = info
        self._sess_options = sess_options
        self._session = None
        self._input = None
        self._size = info.size

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        log.info("Cargando face parser %s", self._info.key)
        self._session = ort.InferenceSession(
            self._path, sess_options=self._sess_options, providers=self._providers
        )
        self._input = self._session.get_inputs()[0]
        shape = self._input.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4 and isinstance(shape[2], int):
            self._size = int(shape[2])

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _labels(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, (self._size, self._size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        blob = rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)
        out = self._session.run(None, {self._input.name: blob})[0]
        # out: (1, C, H, W) -> argmax sobre canales
        return np.argmax(out[0], axis=0).astype(np.int32)

    def region_masks(self, face_bgr: np.ndarray, blur: float = 0.06) -> Dict[str, np.ndarray]:
        """Devuelve máscaras float (0..1) 'face', 'eyes', 'mouth' al tamaño de entrada."""
        if self._session is None:
            self.load()
        h, w = face_bgr.shape[:2]
        labels = self._labels(face_bgr)

        def mask_of(classes):
            m = np.isin(labels, classes).astype(np.float32)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            k = max(1, int(min(h, w) * blur)) | 1
            return np.clip(cv2.GaussianBlur(m, (k, k), 0), 0, 1)

        return {
            "face": mask_of(_FACE_CLASSES),
            "eyes": mask_of(_EYES),
            "mouth": mask_of(_MOUTH),
        }

    def unload(self) -> None:
        self._session = None
