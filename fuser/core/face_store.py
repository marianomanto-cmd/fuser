"""Gestión de las caras fuente (multi-referencia) y selección de objetivos.

Multi-referencia robusta (clave para videos musicales con mucho movimiento):
en lugar de promediar a ciegas los embeddings de todas las fotos, se construye
un **embedding de identidad robusto**:

1. Se toma la cara más grande de cada foto de origen.
2. Se ponderan las referencias por **frontalidad** (las frontales son más
   fiables) y por la confianza del detector.
3. Se **rechazan outliers** (una foto equivocada o de otra persona).
4. Opcionalmente se limita a las **mejores N** referencias.

El embedding resultante es un único vector de 512-d → **no cuesta VRAM** por más
fotos que se usen (lo caro es la detección, que ocurre una sola vez al cargar).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

from .. import config
from ..models.face_analyser import FaceAnalyser
from ..utils.logging import get_logger

log = get_logger(__name__)


def _embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class SourceStats:
    n_input: int
    n_used: int
    mean_yaw: float
    rejected: int

    def summary(self) -> str:
        return (
            f"{self.n_used}/{self.n_input} referencias usadas · "
            f"yaw medio {self.mean_yaw:.0f}° · {self.rejected} descartadas"
        )


class FaceStore:
    def __init__(self, analyser: FaceAnalyser, settings: config.Settings):
        self.analyser = analyser
        self.settings = settings
        self.source_face = None            # objeto con .normed_embedding (para el swapper)
        self.source_faces: List = []       # caras fuente conservadas (para uso futuro)
        self.stats: Optional[SourceStats] = None
        self.reference_embedding: Optional[np.ndarray] = None

    # ----- Fuente (multi-referencia robusta) -----------------------------------
    def set_source(self, images: List[np.ndarray]) -> SourceStats:
        faces = []
        for img in images:
            detected = self.analyser.get_faces(img)
            if detected:
                faces.append(self.analyser.largest_face(detected))
        if not faces:
            raise ValueError(
                "No se detectó ninguna cara en la(s) imagen(es) fuente. "
                "Usa fotos nítidas, de frente y bien iluminadas."
            )

        n_input = len(faces)
        embs = np.stack([_normalize(f.normed_embedding) for f in faces]).astype(np.float32)

        # Pesos por frontalidad + confianza del detector.
        yaws = np.array([abs(self.analyser.estimate_yaw(f)) for f in faces], dtype=np.float32)
        dets = np.array([float(getattr(f, "det_score", 1.0)) for f in faces], dtype=np.float32)
        weights = np.clip(dets, 0.1, 1.0)
        if self.settings.frontal_weighting:
            weights = weights / (1.0 + yaws / 20.0)

        # Promedio ponderado inicial + rechazo de outliers.
        mean = _normalize(np.average(embs, axis=0, weights=weights))
        sims = embs @ mean
        floor = max(self.settings.multi_ref_min_sim, float(sims.mean() - 1.5 * sims.std()))
        keep = sims >= floor
        if keep.sum() == 0:
            keep = np.ones(len(faces), dtype=bool)  # no descartar todo
        rejected = int((~keep).sum())

        kept_faces = [f for f, k in zip(faces, keep) if k]
        kept_embs = embs[keep]
        kept_weights = weights[keep]
        kept_yaws = yaws[keep]

        # Limitar a las mejores N referencias (por peso).
        n_target = self.settings.reference_count
        if n_target and len(kept_faces) > n_target:
            order = np.argsort(-kept_weights)[:n_target]
            kept_faces = [kept_faces[i] for i in order]
            kept_embs = kept_embs[order]
            kept_weights = kept_weights[order]
            kept_yaws = kept_yaws[order]

        final = _normalize(np.average(kept_embs, axis=0, weights=kept_weights))
        self.source_face = SimpleNamespace(normed_embedding=final)
        self.source_faces = kept_faces
        self.stats = SourceStats(
            n_input=n_input, n_used=len(kept_faces),
            mean_yaw=float(kept_yaws.mean()) if len(kept_yaws) else 0.0,
            rejected=rejected,
        )
        log.info("Fuente: %s", self.stats.summary())
        return self.stats

    # ----- Referencia (para selector "reference") ------------------------------
    def set_reference(self, frame: np.ndarray, face_index: int = 0) -> bool:
        faces = self.analyser.get_faces(frame)
        if not faces:
            return False
        idx = max(0, min(face_index, len(faces) - 1))
        self.reference_embedding = faces[idx].normed_embedding
        return True

    # ----- Selección de objetivos ----------------------------------------------
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
                return [self.analyser.largest_face(faces)]
            threshold = self.settings.reference_distance
            return [
                f for f in faces
                # Las caras sintetizadas en huecos de detección no tienen embedding:
                # se omiten del emparejamiento por referencia (sin romper).
                if getattr(f, "normed_embedding", None) is not None
                and _embedding_distance(f.normed_embedding, self.reference_embedding) <= threshold
            ]
        return faces
