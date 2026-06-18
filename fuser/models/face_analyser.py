"""Detección y reconocimiento de caras con InsightFace (buffalo_l).

``buffalo_l`` incluye detección (SCRFD) + embeddings ArcFace, todo en ONNX, y
lo descarga la propia librería InsightFace en ``INSIGHTFACE_ROOT``.

La importación de ``insightface`` es perezosa para que la UI arranque sin la
dependencia instalada.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import INSIGHTFACE_ROOT
from ..utils.logging import get_logger

log = get_logger(__name__)


class FaceAnalyser:
    """Envoltorio de ``insightface.app.FaceAnalysis``."""

    def __init__(self, providers: list, ctx_id: int, det_size: int = 640):
        self._providers = providers
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._app = None

    def load(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis  # import perezoso

        log.info("Cargando detector buffalo_l (det_size=%d, ctx=%d)", self._det_size, self._ctx_id)
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(INSIGHTFACE_ROOT),
            providers=self._providers,
        )
        app.prepare(ctx_id=self._ctx_id, det_size=(self._det_size, self._det_size))
        self._app = app

    @property
    def loaded(self) -> bool:
        return self._app is not None

    def get_faces(self, image: np.ndarray) -> List:
        """Devuelve las caras detectadas ordenadas de izquierda a derecha."""
        if self._app is None:
            self.load()
        faces = self._app.get(image)
        faces.sort(key=lambda f: float(f.bbox[0]))
        return faces

    @staticmethod
    def largest_face(faces: List):
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @staticmethod
    def average_embedding(faces: List) -> np.ndarray:
        """Promedia y normaliza embeddings de varias caras fuente."""
        if not faces:
            raise ValueError("No hay caras para promediar.")
        emb = np.mean([f.normed_embedding for f in faces], axis=0)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    @staticmethod
    def similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Similitud coseno entre embeddings normalizados (1 = idénticos)."""
        return float(np.dot(emb_a, emb_b))

    @staticmethod
    def estimate_yaw(face) -> float:
        """Estima el yaw (giro horizontal) en grados aproximados a partir de los kps.

        Proxy barato: desplazamiento horizontal de la nariz respecto al centro de
        los ojos, normalizado por la distancia interocular. ~0° = frontal; el
        signo indica el lado. Suficiente para ponderar referencias frontales.
        """
        try:
            kps = np.asarray(face.kps, dtype=np.float32)
            eye_c = (kps[0] + kps[1]) / 2.0
            inter = float(np.linalg.norm(kps[1] - kps[0])) + 1e-3
            offset = (kps[2][0] - eye_c[0]) / inter
            return float(np.clip(offset * 90.0, -90.0, 90.0))
        except Exception:
            return 0.0

    def unload(self) -> None:
        self._app = None
