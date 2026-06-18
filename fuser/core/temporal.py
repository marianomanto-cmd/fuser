"""Suavizado temporal de detecciones para vídeo.

El detector de caras se ejecuta de forma independiente en cada frame, lo que
produce un ligero "temblor" (jitter) en los landmarks entre frames consecutivos.
Como el swapper alinea la cara usando esos 5 puntos (kps), el jitter se traduce
en bordes que vibran.

Solución ligera: un tracker que empareja caras entre frames por cercanía de
centroide y aplica un promedio exponencial (EMA) a los landmarks:

    kps_suavizado = alpha * kps_previo + (1 - alpha) * kps_actual

No requiere modelos extra ni VRAM. ``alpha`` controla cuánta memoria temporal
se aplica (0 = sin suavizado, ~0.9 = muy estable pero con "lag").
"""
from __future__ import annotations

from typing import List

import numpy as np


class _Track:
    __slots__ = ("center", "kps", "ttl")

    def __init__(self, center: np.ndarray, kps: np.ndarray):
        self.center = center
        self.kps = kps
        self.ttl = 0


class TemporalSmoother:
    def __init__(self, alpha: float = 0.55, max_rel_dist: float = 0.12, max_ttl: int = 8):
        self.alpha = float(np.clip(alpha, 0.0, 0.95))
        self.max_rel_dist = max_rel_dist
        self.max_ttl = max_ttl
        self._tracks: List[_Track] = []

    def reset(self) -> None:
        self._tracks = []

    @staticmethod
    def _center(bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)

    def smooth(self, faces: List, frame_shape) -> List:
        """Aplica EMA a los kps de cada cara emparejándola con el frame previo."""
        if self.alpha <= 0 or not faces:
            return faces

        diag = float(np.hypot(frame_shape[0], frame_shape[1])) or 1.0
        used = set()

        for face in faces:
            center = self._center(face.bbox)
            best_i, best_d = -1, 1e9
            for i, tr in enumerate(self._tracks):
                if i in used:
                    continue
                d = float(np.linalg.norm(center - tr.center)) / diag
                if d < best_d:
                    best_d, best_i = d, i

            if best_i >= 0 and best_d <= self.max_rel_dist:
                tr = self._tracks[best_i]
                smoothed = self.alpha * tr.kps + (1.0 - self.alpha) * face.kps
                face.kps = smoothed.astype(np.float32)
                tr.kps = face.kps
                tr.center = center
                tr.ttl = 0
                used.add(best_i)
            else:
                self._tracks.append(_Track(center, face.kps.astype(np.float32).copy()))
                used.add(len(self._tracks) - 1)

        # Caduca tracks no emparejados.
        survivors = []
        for i, tr in enumerate(self._tracks):
            if i in used:
                survivors.append(tr)
            else:
                tr.ttl += 1
                if tr.ttl <= self.max_ttl:
                    survivors.append(tr)
        self._tracks = survivors
        return faces
