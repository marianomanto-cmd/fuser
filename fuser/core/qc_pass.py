"""Segunda pasada de control de calidad (QC): detecta frames defectuosos de un
vídeo ya swapeado y los CORRIGE con el MISMO modelo.

Filosofía **auto-referencial**: no usamos umbrales fijos, sino la distribución del
propio vídeo (mediana + MAD, z-score modificado) más comparación con el ORIGINAL,
para no confundir un defecto del swap con un frame legítimamente movido/borroso.

Señales por frame (todo con buffalo_l + OpenCV; el modelo de swap NO se carga aquí):
  - **sin_cara**: la salida no tiene cara pero el original SÍ (el swap se "cayó").
  - **borroso**: la nitidez de la cara SWAPEADA cae muy por debajo de la del mismo
    frame ORIGINAL (si el original ya está movido, no es defecto → no se marca).
  - **identidad**: el embedding ArcFace se aleja de la identidad mediana del vídeo
    (por umbral MAD global y por Hampel para "pops" de un solo frame).
  - **salto**: la caja de la cara pega un brinco puntual respecto a los vecinos.
  - **baja_conf**: la cara swapeada apenas se detecta (artefacto/deformación).

Corrección (mismo modelo):
  1. **Re-swap de recuperación**: reprocesa el frame con detección agresiva y
     enhancer fuerte; se RE-PUNTÚA y solo se queda si mejora (nunca empeora).
  2. **Relleno temporal**: transporta la cara buena del vecino más cercano con flujo
     óptico DIS (cara del mismo modelo en un frame bueno, "arrastrada" por el
     movimiento real) y la compone con máscara suave (nunca seamlessClone).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

ProgressCb = Optional[Callable[[float, str], None]]


@dataclass
class FrameQC:
    idx: int
    has_face: bool = False
    det: float = 0.0
    sharp: float = 0.0
    embed: Optional[np.ndarray] = None
    box: Optional[np.ndarray] = None       # xyxy (salida)
    orig_face: bool = False
    orig_sharp: float = 0.0


@dataclass
class QCReport:
    total: int = 0
    defects: Dict[int, List[str]] = field(default_factory=dict)
    corrected: Dict[int, str] = field(default_factory=dict)
    failed: List[int] = field(default_factory=list)

    def summary(self) -> str:
        by: Dict[str, int] = {}
        for reasons in self.defects.values():
            for r in reasons:
                by[r] = by.get(r, 0) + 1
        methods: Dict[str, int] = {}
        for m in self.corrected.values():
            methods[m] = methods.get(m, 0) + 1
        parts = [f"{self.total} frames · {len(self.defects)} defectuosos"]
        if by:
            parts.append("(" + ", ".join(f"{k}:{v}" for k, v in sorted(by.items())) + ")")
        if self.corrected:
            parts.append("· corregidos " + ", ".join(f"{k}:{v}" for k, v in methods.items()))
        if self.failed:
            parts.append(f"· sin arreglar {len(self.failed)}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Métricas por frame
# ---------------------------------------------------------------------------
def _largest_face(analyser, frame):
    faces = analyser.get_faces(frame)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _sharpness(frame, box, inset: float = 0.12, size: int = 256) -> float:
    """Varianza del Laplaciano en la cara, con recorte INTERIOR (para no medir el
    borde/costura) y NORMALIZADA a un tamaño fijo (invariante a la resolución, para
    comparar swap vs original de forma justa aunque tengan distinto tamaño)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    ix, iy = int(w * inset), int(h * inset)
    x1, y1, x2, y2 = max(0, x1 + ix), max(0, y1 + iy), x2 - ix, y2 - iy
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return 0.0
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def analyze(orig_frames: List[np.ndarray], swap_frames: List[np.ndarray],
            analyser, progress: ProgressCb = None) -> List[FrameQC]:
    """Detecta la cara en cada frame de SALIDA (métricas) y en el ORIGINAL (para
    distinguir defecto del swap de un frame legítimamente movido/borroso)."""
    out: List[FrameQC] = []
    n = len(swap_frames)
    for i in range(n):
        m = FrameQC(idx=i)
        sf = _largest_face(analyser, swap_frames[i])
        if sf is not None:
            emb = getattr(sf, "normed_embedding", None)
            m.has_face = True
            m.det = float(getattr(sf, "det_score", 0.0))
            m.sharp = _sharpness(swap_frames[i], sf.bbox)
            m.embed = np.asarray(emb, np.float32) if emb is not None else None
            m.box = np.asarray(sf.bbox, np.float32)
        if i < len(orig_frames):
            of = _largest_face(analyser, orig_frames[i])
            if of is not None:
                m.orig_face = True
                m.orig_sharp = _sharpness(orig_frames[i], of.bbox)
        out.append(m)
        if progress and (i % 15 == 0 or i == n - 1):
            progress(i / max(1, n), f"QC: analizando {i + 1}/{n}")
    return out


def _robust_floor(values: np.ndarray, k: float) -> float:
    m = float(np.median(values))
    mad = float(np.median(np.abs(values - m)))
    if mad < 1e-9:
        s = float(np.std(values)) + 1e-9
        return m - k * s
    return m - k * 1.4826 * mad


def _hampel_drops(idx_list: List[int], values: List[float], win: int = 3, h: float = 3.0):
    """Índices que son 'pops' HACIA ABAJO (peor) respecto a su ventana local."""
    vals = np.asarray(values, np.float64)
    n = len(vals)
    drops = set()
    for j in range(n):
        lo, hi = max(0, j - win), min(n, j + win + 1)
        w = vals[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) + 1e-9
        if vals[j] < med and abs(vals[j] - med) > h * 1.4826 * mad:
            drops.add(idx_list[j])
    return drops


def flag_defects(metrics: List[FrameQC], sensitivity: float = 0.5) -> Dict[int, List[str]]:
    """Marca frames defectuosos de forma AUTO-REFERENCIAL + comparando con el original."""
    faces = [m for m in metrics if m.has_face and m.embed is not None]
    face_idx = {m.idx for m in metrics if m.has_face}
    defects: Dict[int, List[str]] = {}
    if len(faces) < 6:
        return defects

    sens = float(np.clip(sensitivity, 0.0, 1.0))
    k = 3.5 - 2.5 * sens                                # 3.5 (pocos) … 1.0 (muchos)
    h = 4.0 - 1.5 * sens                                # Hampel: 4.0 … 2.5
    blur_ratio = 0.50 + 0.20 * sens                    # swap < ratio·original -> borroso

    embeds = np.stack([m.embed for m in faces])
    med_embed = np.median(embeds, axis=0)
    med_embed /= (np.linalg.norm(med_embed) + 1e-9)
    sims = (embeds @ med_embed).astype(np.float64)
    sim_floor = _robust_floor(sims, k)
    median_sim = float(np.median(sims))
    # margen ABSOLUTO extra para "identidad": exige una caída real (no solo estadística)
    # -> reduce falsos positivos en frames casi buenos. Menos margen si subís sensibilidad.
    sim_margin = 0.06 - 0.03 * sens
    sim_map = {m.idx: float(s) for m, s in zip(faces, sims)}
    sim_drops = _hampel_drops([m.idx for m in faces], list(sims), h=h)

    dets = [m.det for m in faces]
    det_drops = _hampel_drops([m.idx for m in faces], dets, h=h)

    centers = {m.idx: np.array([(m.box[0] + m.box[2]) / 2, (m.box[1] + m.box[3]) / 2])
               for m in metrics if m.has_face and m.box is not None}
    sizes = {m.idx: max(1.0, float(m.box[2] - m.box[0]))
             for m in metrics if m.has_face and m.box is not None}

    for m in metrics:
        i, reasons = m.idx, []
        if not m.has_face:
            # sin cara: el original tenía cara (o los vecinos la tienen) -> el swap se cayó
            if m.orig_face or (i - 1) in face_idx or (i + 1) in face_idx:
                reasons.append("sin_cara")
        else:
            # borroso: SOLO si el swap está más borroso que el ORIGINAL de ese frame
            # (si el original ya está movido, no es defecto del swap).
            if m.orig_sharp > 5.0 and m.sharp < blur_ratio * m.orig_sharp:
                reasons.append("borroso")
            # identidad: por debajo del suelo MAD (o pop Hampel) Y con caída ABSOLUTA
            # real respecto a la mediana (evita marcar frames casi buenos).
            si = sim_map.get(i, 1.0)
            if (si < sim_floor or i in sim_drops) and si < median_sim - sim_margin:
                reasons.append("identidad")
            # baja confianza de detección puntual (cara deformada/artefacto).
            if (m.det < 0.55 and i in det_drops):
                reasons.append("baja_conf")
            # salto temporal: el centro brinca respecto al vecino previo y NO es un
            # movimiento suave (el frame siguiente vuelve cerca del previo).
            if (i - 1) in centers and (i + 1) in centers and i in centers:
                jump = np.linalg.norm(centers[i] - centers[i - 1]) / sizes[i]
                span = np.linalg.norm(centers[i + 1] - centers[i - 1]) / sizes[i]
                if jump > (0.32 + 0.20 * (1 - sens)) and jump > 3.0 * (span + 1e-3):
                    reasons.append("salto")
        if reasons:
            defects[i] = reasons
    return defects


# ---------------------------------------------------------------------------
# Corrección
# ---------------------------------------------------------------------------
def _feather_mask(shape, box, feather: float = 0.28) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    mask = np.zeros((h, w), np.float32)
    if x2 <= x1 or y2 <= y1:
        return mask
    mask[y1:y2, x1:x2] = 1.0
    fw = int(max(5, (x2 - x1) * feather))
    if fw % 2 == 0:
        fw += 1
    return cv2.GaussianBlur(mask, (fw, fw), 0)


def _optical_flow_fill(cur_orig, good_orig, good_swap, box) -> Optional[np.ndarray]:
    """Transporta la cara de ``good_swap`` al frame actual con flujo óptico DIS.

    Flujo del ACTUAL -> BUENO (para que ``remap`` traiga los píxeles del bueno a la
    geometría del actual), luego se compone la región de la cara con máscara suave.
    """
    try:
        g_cur = cv2.cvtColor(cur_orig, cv2.COLOR_BGR2GRAY)
        g_good = cv2.cvtColor(good_orig, cv2.COLOR_BGR2GRAY)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(g_cur, g_good, None)               # actual -> bueno
        h, w = cur_orig.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(good_swap, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = _feather_mask(cur_orig.shape, box)[:, :, None]
        out = warped.astype(np.float32) * mask + cur_orig.astype(np.float32) * (1 - mask)
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception as exc:  # pragma: no cover
        log.warning("Relleno por flujo óptico falló: %s", exc)
        return None


def correct(orig_frames: List[np.ndarray], swap_frames: List[np.ndarray],
            metrics: List[FrameQC], defects: Dict[int, List[str]], analyser, *,
            reswap_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            progress: ProgressCb = None) -> QCReport:
    """Corrige IN-PLACE los frames marcados sobre ``swap_frames``. Devuelve el reporte."""
    report = QCReport(total=len(swap_frames), defects=dict(defects))
    if not defects:
        return report
    good = sorted(m.idx for m in metrics if m.has_face and m.idx not in defects)
    med_sharp = float(np.median([m.sharp for m in metrics if m.has_face] or [1.0]))

    def nearest_good(i: int) -> Optional[int]:
        best, bestd = None, 10 ** 9
        for g in good:
            d = abs(g - i)
            if d < bestd:
                best, bestd = g, d
        return best

    order = sorted(defects)
    for n, i in enumerate(order):
        reasons = defects[i]
        fixed = False
        # 1) Re-swap de recuperación (mismo modelo, detección agresiva) + guardrail.
        if reswap_fn is not None:
            try:
                cand = reswap_fn(orig_frames[i])
                cf = _largest_face(analyser, cand)
                if cf is not None:
                    sh = _sharpness(cand, cf.bbox)
                    # Se queda solo si MEJORA: recupera cara caída, o queda razonablemente
                    # nítido (no peor que ~0.6× la mediana del vídeo) y mejor que el actual.
                    better = ("sin_cara" in reasons) or (sh >= 0.6 * med_sharp and sh >= metrics[i].sharp)
                    if better:
                        swap_frames[i] = cand
                        report.corrected[i] = "reswap"
                        fixed = True
            except Exception as exc:  # pragma: no cover
                log.warning("Re-swap de recuperación falló en frame %d: %s", i, exc)
        # 2) Relleno temporal desde el vecino bueno más cercano.
        if not fixed:
            g = nearest_good(i)
            box = metrics[i].box if metrics[i].box is not None else (metrics[g].box if g is not None else None)
            if g is not None and box is not None:
                filled = _optical_flow_fill(orig_frames[i], orig_frames[g], swap_frames[g], box)
                if filled is not None:
                    swap_frames[i] = filled
                    report.corrected[i] = "relleno_temporal"
                    fixed = True
        if not fixed:
            report.failed.append(i)
        if progress:
            progress(n / max(1, len(order)), f"QC: corrigiendo {n + 1}/{len(order)}")
    return report


# ---------------------------------------------------------------------------
# CLI de prueba (solo análisis: no reprocesa)
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuser.config import Settings
    from fuser.core.memory_manager import MemoryManager
    from fuser.models.face_analyser import FaceAnalyser
    from fuser.utils import video as videoutil

    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--swapped", required=True)
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()

    mm = MemoryManager(Settings())
    det = FaceAnalyser(mm.analyser_providers(), mm.ctx_id(), mm.det_size)
    det.load()
    orig = list(videoutil.read_frames(a.orig))
    swap = list(videoutil.read_frames(a.swapped))
    if a.max:
        orig, swap = orig[:a.max], swap[:a.max]
    # Alinea el original a la MISMA resolución que la salida (como en el pipeline
    # real, donde el QC compara la salida con el frame de entrada ya redimensionado).
    orig = [cv2.resize(o, (swap[i].shape[1], swap[i].shape[0])) if i < len(swap) else o
            for i, o in enumerate(orig)]
    metrics = analyze(orig, swap, det, progress=lambda f, m="": None)
    defects = flag_defects(metrics, a.sensitivity)
    n_face = sum(1 for m in metrics if m.has_face)
    print(f"frames={len(swap)} con_cara={n_face} defectuosos={len(defects)} (sens={a.sensitivity})")
    for i in sorted(defects):
        print(f"  f{i}: {', '.join(defects[i])}")


if __name__ == "__main__":
    _main()
