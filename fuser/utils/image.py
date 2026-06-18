"""Operaciones de imagen: máscaras, paste-back afín, color matching, resize.

Aquí vive toda la "costura" entre la cara generada (alineada) y el frame
original. Un buen paste-back con máscara suavizada es lo que diferencia un
swap con bordes visibles de uno limpio.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _odd(n: int) -> int:
    n = int(round(n))
    return n + 1 if n % 2 == 0 else max(1, n)


def build_soft_mask(
    height: int,
    width: int,
    blur: float = 0.25,
    padding: float = 0.0,
) -> np.ndarray:
    """Crea una máscara float32 (0..1) con bordes suavizados.

    - ``padding`` recorta la máscara hacia dentro (0..1 del lado menor),
      útil para que el swap no invada pelo/frente.
    - ``blur`` controla el plumeado del borde (0..1 del lado menor).
    """
    mask = np.zeros((height, width), dtype=np.float32)
    side = min(height, width)
    pad = int(side * float(np.clip(padding, 0.0, 0.45)))
    mask[pad : height - pad, pad : width - pad] = 1.0

    blur_px = int(side * float(np.clip(blur, 0.0, 0.9)))
    if blur_px > 0:
        k = _odd(blur_px)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)


def paste_back(
    frame: np.ndarray,
    face: np.ndarray,
    affine_matrix: np.ndarray,
    mask_blur: float = 0.25,
    mask_padding: float = 0.0,
    opacity: float = 1.0,
) -> np.ndarray:
    """Pega ``face`` (cara alineada) en ``frame`` invirtiendo la matriz afín.

    ``affine_matrix`` es la matriz 2x3 que mapea el frame original -> recorte
    alineado (la que devuelve InSwapper). Se invierte para volver al frame.
    """
    h, w = frame.shape[:2]
    fh, fw = face.shape[:2]

    inverse = cv2.invertAffineTransform(affine_matrix)
    mask = build_soft_mask(fh, fw, blur=mask_blur, padding=mask_padding)

    warped_face = cv2.warpAffine(
        face, inverse, (w, h), borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)
    warped_mask = cv2.warpAffine(mask, inverse, (w, h))
    warped_mask = np.clip(warped_mask * float(np.clip(opacity, 0.0, 1.0)), 0.0, 1.0)
    warped_mask = warped_mask[:, :, None]

    out = warped_face * warped_mask + frame.astype(np.float32) * (1.0 - warped_mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Transferencia de color Reinhard (en espacio LAB) de ``target`` a ``source``.

    Ajusta media y desviación de cada canal LAB para que ``source`` (la cara
    generada) adopte la iluminación/tono de ``target`` (la cara original).
    """
    src = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = src.mean(axis=(0, 1)), src.std(axis=(0, 1)) + 1e-6
    tgt_mean, tgt_std = tgt.mean(axis=(0, 1)), tgt.std(axis=(0, 1)) + 1e-6

    src = (src - src_mean) / src_std * tgt_std + tgt_mean
    src = np.clip(src, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src, cv2.COLOR_LAB2BGR)


def scale_affine(affine_matrix: np.ndarray, scale: float) -> np.ndarray:
    """Escala una matriz afín 2x3 para un recorte ``scale`` veces mayor.

    Si ``p_crop = M @ [x, y, 1]``, entonces para un recorte ``scale`` veces más
    grande basta con ``scale * M`` (escala coords de salida). Permite pegar la
    cara realzada a 512 px usando la matriz original calculada a 128 px.
    """
    return (affine_matrix.astype(np.float32) * float(scale)).astype(np.float32)


def limit_resolution(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Reduce un frame si su lado mayor supera ``max_side``.

    Devuelve ``(frame_reducido, factor)`` donde ``factor`` es la escala aplicada
    (1.0 si no se tocó). Procesar a menor resolución ahorra VRAM y acelera.
    """
    if max_side <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame, 1.0
    factor = max_side / float(longest)
    resized = cv2.resize(
        frame, (int(round(w * factor)), int(round(h * factor))), interpolation=cv2.INTER_AREA
    )
    return resized, factor


def to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
