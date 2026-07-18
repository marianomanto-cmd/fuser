"""Post-procesado fotométrico del swap (anti "cara pegada").

El paso que más delata un swap no suele ser la identidad sino la FOTOMETRÍA:
la cara nueva llega con el tono/contraste del modelo, no con la luz de la
escena. Este módulo armoniza el color/iluminación de la zona swapeada contra
el frame ORIGINAL, de forma agnóstica al motor y sin modelos extra:

1. ``paste_diff_mask``  — detecta la huella EXACTA del pegado comparando el
   frame antes/después del swap (nada de detectores: la diferencia ES la zona
   pegada), con limpieza morfológica y borde plumado.
2. ``harmonize_swap``   — transferencia de estadísticas en LAB (media/desvío
   por canal, estilo Reinhard) de la zona original a la swapeada, con ganancia
   de contraste acotada y mezcla SOLO dentro de la máscara.

Todo numpy/cv2 en CPU: coste por frame ~ms, cero VRAM. Lo usa el pipeline
cuando ``Settings.color_harmonize`` está activo (modo 🚀 MAXIMUM SWAP).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def paste_diff_mask(
    before: np.ndarray,
    after: np.ndarray,
    threshold: int = 6,
    feather: int = 31,
) -> Optional[np.ndarray]:
    """Máscara float [0..1] de la zona que el swap modificó (la cara pegada).

    Devuelve None si el frame no cambió (sin cara detectada / swap sin efecto),
    para que el llamador lo deje pasar sin coste.
    """
    if before.shape != after.shape:
        return None
    diff = cv2.absdiff(before, after).max(axis=2)
    mask = (diff > threshold).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return None
    # Cierra huecos (ojos/dientes con poca diferencia), quita motas sueltas y
    # expande un pelo para cubrir el borde del blending del motor.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(5))
    mask = cv2.dilate(mask, _kernel(7))
    feather = max(3, int(feather)) | 1
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask.astype(np.float32) / 255.0


def _masked_stats(channel: np.ndarray, weights: np.ndarray) -> tuple:
    total = float(weights.sum())
    mean = float((channel * weights).sum() / total)
    var = float((weights * (channel - mean) ** 2).sum() / total)
    return mean, max(var, 1e-6) ** 0.5


def harmonize_swap(
    before: np.ndarray,
    after: np.ndarray,
    strength: float = 0.8,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Iguala tono/iluminación de la zona swapeada a la del frame original.

    Transferencia media/desvío por canal en LAB (Reinhard) restringida a la
    máscara del pegado. La ganancia de contraste se acota a [0.6, 1.6] para no
    "quemar" ni lavar la cara cuando el motor entrega un contraste muy distinto.
    ``strength`` regula la mezcla final (0 = nada, 1 = corrección completa).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return after
    if mask is None:
        mask = paste_diff_mask(before, after)
    if mask is None or float(mask.sum()) < 400.0:  # zona ínfima: no vale la pena
        return after

    lab_b = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Estadísticas con el CORE de la máscara (>0.5): el borde plumado mezclaría
    # fondo en las medias y sesgaría la corrección.
    core = (mask > 0.5).astype(np.float32)
    if float(core.sum()) < 400.0:
        core = mask
    corrected = lab_a.copy()
    for c in range(3):
        mu_b, sd_b = _masked_stats(lab_b[..., c], core)
        mu_a, sd_a = _masked_stats(lab_a[..., c], core)
        gain = float(np.clip(sd_b / sd_a, 0.6, 1.6))
        corrected[..., c] = (lab_a[..., c] - mu_a) * gain + mu_b

    corrected = cv2.cvtColor(
        np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    ).astype(np.float32)
    w = (mask * strength)[..., None]
    out = after.astype(np.float32) * (1.0 - w) + corrected * w
    return np.clip(out, 0, 255).astype(np.uint8)
