"""Gestión de las caras fuente y selección de las caras objetivo.

- ``set_source``: detecta la cara en una o varias imágenes fuente y construye el
  embedding (promediado si hay varias) que usará el swapper.
- ``set_reference``: registra una cara concreta del vídeo para el modo "swap solo
  a la persona de referencia".
- ``select_targets``: dado el conjunto de caras de un frame, decide a cuáles se
  les aplica el swap según el modo elegido.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import numpy as np

from .. import config
from ..models.face_analyser import FaceAnalyser
from ..utils.logging import get_logger

log = get_logger(__name__)


def _embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia euclídea entre embeddings normalizados (menor = más parecido)."""
    return float(np.linalg.norm(a - b))


class FaceStore:
    def __init__(self, analyser: FaceAnalyser, settings: config.Settings):
        self.analyser = analyser
        self.settings = settings
        self.source_face = None
        self.reference_embedding: Optional[np.ndarray] = None

    # ----- Fuente --------------------------------------------------------------
    def set_source(self, images: List[np.ndarray]) -> None:
        faces = []
        for img in images:
            detected = self.analyser.get_faces(img)
            if detected:
                faces.append(self.analyser.largest_face(detected))
        if not faces:
            raise ValueError(
                "No se detectó ninguna cara en la(s) imagen(es) fuente. "
                "Usa una foto nítida, de frente y bien iluminada."
            )
        if self.settings.source_average and len(faces) > 1:
            emb = self.analyser.average_embedding(faces)
            self.source_face = SimpleNamespace(normed_embedding=emb)
            log.info("Embedding fuente promediado de %d caras.", len(faces))
        else:
            self.source_face = faces[0]

    # ----- Referencia ----------------------------------------------------------
    def set_reference(self, frame: np.ndarray, face_index: int = 0) -> bool:
        faces = self.analyser.get_faces(frame)
        if not faces:
            return False
        idx = max(0, min(face_index, len(faces) - 1))
        self.reference_embedding = faces[idx].normed_embedding
        return True

    # ----- Selección de objetivos ---------------------------------------------
    def select_targets(self, faces: List) -> List:
        if not faces:
            return []
        mode = self.settings.face_selector

        if mode == config.FACE_SELECTOR_ALL:
            return faces

        if mode == config.FACE_SELECTOR_LARGEST:
            return [self.analyser.largest_face(faces)]

        if mode == config.FACE_SELECTOR_INDEX:
            idx = self.settings.reference_face_index
            return [faces[idx]] if 0 <= idx < len(faces) else []

        if mode == config.FACE_SELECTOR_REFERENCE:
            if self.reference_embedding is None:
                # Sin referencia explícita, comportamiento seguro: la cara más grande.
                return [self.analyser.largest_face(faces)]
            threshold = self.settings.reference_distance
            matches = [
                f
                for f in faces
                if _embedding_distance(f.normed_embedding, self.reference_embedding) <= threshold
            ]
            return matches

        return faces
