"""Estabilidad temporal de las detecciones para vídeo.

Dos mecanismos complementarios:

1. ``TemporalSmoother`` — suavizado **causal adaptativo al movimiento** (1 pasada).
   El problema del EMA clásico es que "arrastra" (lag) los movimientos rápidos,
   justo lo peor para una **boca cantando**. Aquí el factor de suavizado se
   reduce automáticamente cuando hay movimiento grande: jitter pequeño → suaviza;
   movimiento rápido → responde al instante (sin lag ni fantasmas).

2. ``apply_two_pass_smoothing`` — suavizado **centrado bilateral** (2 pasadas,
   usa RAM). Al tener todos los landmarks de un tramo en RAM, se filtra cada
   "track" con una ventana centrada que es **bilateral en el tiempo**: promedia
   frames vecinos parecidos (quita temblor) pero respeta los cambios bruscos
   (la boca abriéndose). Resultado: máxima estabilidad sin perder expresión.
"""
from __future__ import annotations

from typing import List

import numpy as np


def _inter_eye(kps: np.ndarray) -> float:
    kps = np.asarray(kps, dtype=np.float32)
    if len(kps) >= 2:
        return float(np.linalg.norm(kps[1] - kps[0])) + 1e-3
    return 1.0


# ---------------------------------------------------------------------------
# 1 pasada: EMA causal adaptativo al movimiento
# ---------------------------------------------------------------------------
class _Track:
    __slots__ = ("center", "kps", "ttl")

    def __init__(self, center: np.ndarray, kps: np.ndarray):
        self.center = center
        self.kps = kps
        self.ttl = 0


class TemporalSmoother:
    def __init__(
        self,
        alpha: float = 0.55,
        max_rel_dist: float = 0.12,
        max_ttl: int = 8,
        motion_adaptive: bool = True,
    ):
        self.alpha = float(np.clip(alpha, 0.0, 0.95))
        self.max_rel_dist = max_rel_dist
        self.max_ttl = max_ttl
        self.motion_adaptive = motion_adaptive
        self._tracks: List[_Track] = []

    def reset(self) -> None:
        self._tracks = []

    @staticmethod
    def _center(bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)

    def _effective_alpha(self, cur_kps: np.ndarray, prev_kps: np.ndarray) -> float:
        if not self.motion_adaptive:
            return self.alpha
        disp = float(np.linalg.norm(cur_kps - prev_kps, axis=1).mean())
        rel = disp / _inter_eye(cur_kps)
        # rel pequeño (jitter) -> ~alpha ; rel grande (movimiento) -> ~0
        return self.alpha * float(np.exp(-rel / 0.15))

    def smooth(self, faces: List, frame_shape) -> List:
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
                eff = self._effective_alpha(face.kps, tr.kps)
                smoothed = eff * tr.kps + (1.0 - eff) * face.kps
                face.kps = smoothed.astype(np.float32)
                tr.kps = face.kps
                tr.center = center
                tr.ttl = 0
                used.add(best_i)
            else:
                self._tracks.append(_Track(center, face.kps.astype(np.float32).copy()))
                used.add(len(self._tracks) - 1)

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


# ---------------------------------------------------------------------------
# 2 pasadas: tracking + suavizado centrado bilateral (usa RAM)
# ---------------------------------------------------------------------------
def build_tracks(frames_faces: List[List], max_rel_dist: float = 0.06) -> List[List[dict]]:
    """Agrupa caras en *tracks* a lo largo de los frames por cercanía de centroide.

    ``frames_faces`` es una lista (por frame) de listas de caras (objetos con
    ``.bbox`` y ``.kps``). Devuelve una lista de tracks; cada track es una lista
    de ``{"frame": i, "face": face}`` ordenada por frame.
    """
    tracks: List[List[dict]] = []
    last_centroid: List[np.ndarray] = []
    last_seen: List[int] = []

    def centroid(f):
        b = f.bbox
        return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float32)

    for i, faces in enumerate(frames_faces):
        used = set()
        diag = 1.0
        for f in faces:
            diag = max(diag, float(f.bbox[2] - f.bbox[0]))
        for f in faces:
            c = centroid(f)
            best, best_d = -1, 1e9
            for ti in range(len(tracks)):
                if ti in used or i - last_seen[ti] > 12:
                    continue
                d = float(np.linalg.norm(c - last_centroid[ti])) / (diag * 6 + 1e-3)
                if d < best_d:
                    best_d, best = d, ti
            if best >= 0 and best_d <= max_rel_dist:
                tracks[best].append({"frame": i, "face": f})
                last_centroid[best] = c
                last_seen[best] = i
                used.add(best)
            else:
                tracks.append([{"frame": i, "face": f}])
                last_centroid.append(c)
                last_seen.append(i)
                used.add(len(tracks) - 1)
    return tracks


def centered_smooth_kps(
    seq: List[np.ndarray],
    time_sigma: float = 2.0,
    range_rel: float = 0.25,
    motion_adaptive: bool = True,
) -> List[np.ndarray]:
    """Filtro temporal centrado y **bilateral** sobre una secuencia de kps.

    - Término temporal: gaussiana sobre la distancia de frames.
    - Término de rango (bilateral): atenúa frames cuyos kps difieren mucho del
      central → preserva los cambios rápidos (boca) y solo promedia el temblor.
    """
    n = len(seq)
    if n == 0:
        return seq
    W = max(1, int(round(3 * time_sigma)))
    out = []
    for i in range(n):
        ki = seq[i]
        scale = _inter_eye(ki)
        acc = np.zeros_like(ki, dtype=np.float32)
        wsum = 0.0
        for j in range(max(0, i - W), min(n, i + W + 1)):
            wt = np.exp(-((i - j) ** 2) / (2 * time_sigma ** 2))
            if motion_adaptive:
                d = float(np.linalg.norm(seq[j] - ki, axis=1).mean()) / scale
                wr = np.exp(-(d ** 2) / (2 * range_rel ** 2))
            else:
                wr = 1.0
            w = wt * wr
            acc += w * seq[j]
            wsum += w
        out.append((acc / wsum).astype(np.float32) if wsum > 0 else ki)
    return out


def apply_two_pass_smoothing(
    frames_faces: List[List],
    time_sigma: float = 2.0,
    motion_adaptive: bool = True,
) -> int:
    """Suaviza in-place los kps de todas las caras agrupándolas en tracks.

    Devuelve el número de tracks suavizados. Pensado para ejecutarse sobre un
    tramo de frames almacenado en RAM antes de renderizar (2.ª pasada).
    """
    tracks = build_tracks(frames_faces)
    for track in tracks:
        if len(track) < 3:
            continue
        seq = [np.asarray(item["face"].kps, dtype=np.float32) for item in track]
        smoothed = centered_smooth_kps(seq, time_sigma=time_sigma, motion_adaptive=motion_adaptive)
        for item, sk in zip(track, smoothed):
            item["face"].kps = sk
    return len(tracks)
