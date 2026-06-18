"""Operaciones de imagen: máscaras por regiones, paste-back afín, realce local.

Aquí vive toda la "costura" entre la cara generada (alineada) y el frame
original. La calidad fina del swap depende casi por completo de esta etapa:

- Una **máscara que sigue el contorno real de la cara** (casco convexo de los
  landmarks) en lugar de un rectángulo: evita pegar piel sobre orejas, pelo o
  fondo en **perfiles laterales**.
- **Máscaras de región** (ojos / boca) para preservar y realzar localmente esas
  zonas críticas (ojos vivos, dientes nítidos al cantar).
- **Realce local** (unsharp enmascarado) para devolver detalle a ojos y dientes
  sin pasar por modelos extra ni aplanar la cara.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _odd(n: int) -> int:
    n = int(round(n))
    return n + 1 if n % 2 == 0 else max(1, n)


def transform_points(points: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    """Aplica una matriz afín 2x3 a un conjunto de puntos Nx2."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homog = np.hstack([pts, ones])
    return homog @ affine_matrix.T.astype(np.float32)


def _feather(mask: np.ndarray, blur: float) -> np.ndarray:
    side = min(mask.shape[:2])
    blur_px = int(side * float(np.clip(blur, 0.0, 0.9)))
    if blur_px > 0:
        k = _odd(blur_px)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)


def build_soft_mask(height: int, width: int, blur: float = 0.25, padding: float = 0.0) -> np.ndarray:
    """Máscara rectangular float32 (0..1) con bordes suavizados (fallback básico)."""
    mask = np.zeros((height, width), dtype=np.float32)
    side = min(height, width)
    pad = int(side * float(np.clip(padding, 0.0, 0.45)))
    mask[pad : height - pad, pad : width - pad] = 1.0
    return _feather(mask, blur)


def convex_hull_mask(
    points_aligned: np.ndarray,
    size: int,
    blur: float = 0.25,
    padding: float = 0.0,
) -> np.ndarray:
    """Máscara del casco convexo de los landmarks (en espacio alineado de ``size``).

    Sigue el contorno real de la cara → en perfiles no invade oreja/pelo/fondo.
    ``padding`` encoge el casco hacia el centroide (recorta borde).
    """
    pts = np.asarray(points_aligned, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((size, size), dtype=np.float32)
    if len(pts) < 3:
        return build_soft_mask(size, size, blur, padding)

    hull = cv2.convexHull(pts.astype(np.float32))
    if padding > 0:
        center = pts.mean(axis=0, keepdims=True)
        shrink = float(np.clip(1.0 - padding, 0.5, 1.0))
        hull = ((hull.reshape(-1, 2) - center) * shrink + center).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1.0)
    return _feather(mask, blur)


def ellipse_face_mask(
    kps_aligned: np.ndarray,
    size: int,
    blur: float = 0.25,
    padding: float = 0.0,
    expand: float = 1.55,
) -> np.ndarray:
    """Máscara elíptica de la cara a partir de los 5 kps (fallback robusto).

    Útil cuando no hay landmarks de 106 puntos. La elipse se orienta según la
    línea de los ojos, por lo que también ayuda en tomas inclinadas.
    """
    kps = np.asarray(kps_aligned, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((size, size), dtype=np.float32)
    if len(kps) < 5:
        return build_soft_mask(size, size, blur, padding)

    eye_l, eye_r, nose, mouth_l, mouth_r = kps[0], kps[1], kps[2], kps[3], kps[4]
    eye_c = (eye_l + eye_r) / 2.0
    mouth_c = (mouth_l + mouth_r) / 2.0
    center = (eye_c + mouth_c) / 2.0
    # Ejes a partir de la geometría facial.
    inter_eye = float(np.linalg.norm(eye_r - eye_l)) + 1e-3
    vertical = float(np.linalg.norm(mouth_c - eye_c)) + 1e-3
    ax = inter_eye * expand * (1.0 - 0.5 * float(np.clip(padding, 0, 0.4)))
    ay = vertical * 1.7 * expand * (1.0 - 0.5 * float(np.clip(padding, 0, 0.4)))
    angle = np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))
    # Centro ligeramente bajado para cubrir mentón.
    center = center + (mouth_c - eye_c) * 0.15
    cv2.ellipse(
        mask, (int(center[0]), int(center[1])), (int(ax), int(ay)),
        float(angle), 0, 360, 1.0, -1,
    )
    return _feather(mask, blur)


def eye_mouth_region_masks(
    kps_aligned: np.ndarray, size: int, mouth_open_boost: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (máscara_ojos, máscara_boca) en espacio alineado, suavizadas.

    Derivadas de los 5 kps (fiables en todos los ángulos). La región de boca se
    alarga verticalmente para cubrir la boca **muy abierta** al cantar.
    """
    kps = np.asarray(kps_aligned, dtype=np.float32).reshape(-1, 2)
    eyes = np.zeros((size, size), dtype=np.float32)
    mouth = np.zeros((size, size), dtype=np.float32)
    if len(kps) < 5:
        return eyes, mouth

    eye_l, eye_r, nose, mouth_l, mouth_r = kps[0], kps[1], kps[2], kps[3], kps[4]
    inter_eye = float(np.linalg.norm(eye_r - eye_l)) + 1e-3
    angle = float(np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])))

    # --- Ojos: una elipse por ojo ---
    eye_ax = int(inter_eye * 0.42)
    eye_ay = int(inter_eye * 0.26)
    for c in (eye_l, eye_r):
        cv2.ellipse(eyes, (int(c[0]), int(c[1])), (max(2, eye_ax), max(2, eye_ay)),
                    angle, 0, 360, 1.0, -1)

    # --- Boca: elipse alargada hacia abajo (boca abierta) ---
    mouth_c = (mouth_l + mouth_r) / 2.0
    mouth_w = float(np.linalg.norm(mouth_r - mouth_l)) + 1e-3
    m_ax = int(mouth_w * 0.75)
    m_ay = int(mouth_w * 0.55 * (1.0 + 0.8 * float(np.clip(mouth_open_boost, 0, 2))))
    # Bajar el centro: al abrir la boca, el interior/mentón cae.
    down = (mouth_c - (eye_l + eye_r) / 2.0)
    down = down / (np.linalg.norm(down) + 1e-3)
    m_center = mouth_c + down * (m_ay * 0.25)
    cv2.ellipse(mouth, (int(m_center[0]), int(m_center[1])), (max(3, m_ax), max(3, m_ay)),
                angle, 0, 360, 1.0, -1)

    eyes = cv2.GaussianBlur(eyes, (_odd(size * 0.04), _odd(size * 0.04)), 0)
    mouth = cv2.GaussianBlur(mouth, (_odd(size * 0.05), _odd(size * 0.05)), 0)
    return np.clip(eyes, 0, 1), np.clip(mouth, 0, 1)


def frame_eye_mouth_masks(
    kps: np.ndarray, frame_shape, mouth_open_boost: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Máscaras de ojos/boca en coordenadas del **frame completo** (kps en frame).

    Se usa para el post-procesado por regiones aplicado sobre la salida ya
    compuesta (p. ej. la de FaceFusion): realzar ojos y dientes sin tocar el
    resto. El plumeado es relativo a la cara (distancia interocular), no al frame.
    """
    h, w = frame_shape[:2]
    eyes = np.zeros((h, w), dtype=np.float32)
    mouth = np.zeros((h, w), dtype=np.float32)
    kps = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if len(kps) < 5:
        return eyes, mouth

    eye_l, eye_r, nose, mouth_l, mouth_r = kps[0], kps[1], kps[2], kps[3], kps[4]
    inter = float(np.linalg.norm(eye_r - eye_l)) + 1e-3
    angle = float(np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])))

    eye_ax, eye_ay = int(inter * 0.42), int(inter * 0.26)
    for c in (eye_l, eye_r):
        cv2.ellipse(eyes, (int(c[0]), int(c[1])), (max(2, eye_ax), max(2, eye_ay)),
                    angle, 0, 360, 1.0, -1)

    mouth_c = (mouth_l + mouth_r) / 2.0
    mouth_w = float(np.linalg.norm(mouth_r - mouth_l)) + 1e-3
    m_ax = int(mouth_w * 0.75)
    m_ay = int(mouth_w * 0.55 * (1.0 + 0.8 * float(np.clip(mouth_open_boost, 0, 2))))
    down = mouth_c - (eye_l + eye_r) / 2.0
    down = down / (np.linalg.norm(down) + 1e-3)
    m_center = mouth_c + down * (m_ay * 0.25)
    cv2.ellipse(mouth, (int(m_center[0]), int(m_center[1])), (max(3, m_ax), max(3, m_ay)),
                angle, 0, 360, 1.0, -1)

    ke = _odd(max(3, inter * 0.30))
    km = _odd(max(3, mouth_w * 0.35))
    eyes = cv2.GaussianBlur(eyes, (ke, ke), 0)
    mouth = cv2.GaussianBlur(mouth, (km, km), 0)
    return np.clip(eyes, 0, 1), np.clip(mouth, 0, 1)


def mouth_aspect_ratio(kps, landmark_106=None) -> Optional[float]:
    """MAR (mouth aspect ratio) a partir de los 106 landmarks, **sin índices fijos**.

    Mide la extensión **vertical** de los puntos que caen en la banda de la boca
    respecto al **ancho** de la boca. Robusto a la orientación (usa los ejes de la
    boca). Devuelve ``None`` si no hay 106 landmarks. MAR bajo = boca cerrada;
    MAR alto = boca muy abierta (cantando).
    """
    if landmark_106 is None:
        return None
    pts = np.asarray(landmark_106, dtype=np.float32).reshape(-1, 2)
    k = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if len(k) < 5 or len(pts) < 20:
        return None
    mouth_l, mouth_r = k[3], k[4]
    mc = (mouth_l + mouth_r) / 2.0
    axis = mouth_r - mouth_l
    w = float(np.linalg.norm(axis)) + 1e-3
    ux = axis / w                       # eje horizontal de la boca
    uy = np.array([-ux[1], ux[0]], dtype=np.float32)  # eje vertical (perpendicular)
    rel = pts - mc
    horiz = rel @ ux
    vert = rel @ uy
    # Banda ceñida a la boca: excluye nariz (arriba) y mentón (abajo).
    band = (np.abs(horiz) < 0.6 * w) & (np.abs(vert) < 0.5 * w)
    if int(band.sum()) < 6:
        return None
    v = vert[band]
    return float((v.max() - v.min()) / w)


def mouth_openness(kps, landmark_106=None, frame=None, mouth_mask=None) -> float:
    """Apertura de la boca (0..1). Usa MAR de landmarks; si no hay, cae al contraste.

    Es el detector que pide el caso musical: 0 = boca cerrada, 1 = muy abierta.
    """
    mar = mouth_aspect_ratio(kps, landmark_106)
    if mar is not None:
        # MAR ~0.18-0.25 cerrada ; ~0.5+ muy abierta.
        return float(np.clip((mar - 0.22) / 0.33, 0.0, 1.0))
    if frame is not None and mouth_mask is not None:
        return _region_contrast(frame, mouth_mask)
    return 0.5


def _region_contrast(frame: np.ndarray, mask: np.ndarray) -> float:
    """Contraste local (0..1) dentro de la máscara. Alto = boca abierta con dientes.

    Cuando la boca está muy abierta cantando, el interior tiene dientes claros y
    sombra oscura → mucho contraste. Cerrada (labios) → poco. Sirve para aplicar
    el realce de dientes SOLO cuando hace falta.
    """
    sel = mask > 0.4
    if sel.sum() < 16:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    std = float(gray[sel].std())
    return float(np.clip(std / 55.0, 0.0, 1.0))


def frame_face_mask(kps, frame_shape, expand: float = 1.55, blur: float = 0.08) -> np.ndarray:
    """Máscara elíptica de la cara en coordenadas del **frame completo** (desde 5 kps).

    Núcleo alto en el centro y borde suave; útil para mezclar los bordes de la
    cara (mandíbula/oreja) hacia el original en perfiles laterales.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    kps = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if len(kps) < 5:
        return mask
    eye_l, eye_r, nose, mouth_l, mouth_r = kps[0], kps[1], kps[2], kps[3], kps[4]
    eye_c = (eye_l + eye_r) / 2.0
    mouth_c = (mouth_l + mouth_r) / 2.0
    center = (eye_c + mouth_c) / 2.0 + (mouth_c - eye_c) * 0.15
    inter = float(np.linalg.norm(eye_r - eye_l)) + 1e-3
    vert = float(np.linalg.norm(mouth_c - eye_c)) + 1e-3
    ax = int(inter * expand)
    ay = int(vert * 1.7 * expand)
    angle = float(np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])))
    cv2.ellipse(mask, (int(center[0]), int(center[1])), (max(2, ax), max(2, ay)),
                angle, 0, 360, 1.0, -1)
    k = _odd(max(3, inter * blur * 3.0))
    return np.clip(cv2.GaussianBlur(mask, (k, k), 0), 0, 1)


def enhance_regions(
    frame: np.ndarray,
    kps_list,
    eye_strength: float = 0.0,
    mouth_strength: float = 0.0,
    mouth_open_boost: float = 1.0,
    adaptive_mouth: bool = True,
) -> np.ndarray:
    """Realza ojos y boca/dientes en el frame para una o varias caras.

    Post-procesado independiente del motor (ideal para reforzar la salida de
    FaceFusion). Con ``adaptive_mouth`` la fuerza en la boca se **modula por el
    contraste local**: cuando se ven los dientes (boca abierta al cantar) realza
    más; con la boca cerrada no sobre-afila.
    """
    out = frame
    for kps in kps_list:
        eyes, mouth = frame_eye_mouth_masks(kps, out.shape, mouth_open_boost)
        inter = float(np.linalg.norm(np.asarray(kps[1]) - np.asarray(kps[0]))) + 1e-3
        radius = max(3.0, inter * 0.06)  # detalle fino, no desenfoque grande
        if eye_strength > 0 and eyes.max() > 0:
            out = apply_local_detail(out, eyes, amount=eye_strength * 1.2, radius=radius)
        if mouth_strength > 0 and mouth.max() > 0:
            m_str = mouth_strength
            if adaptive_mouth:
                openness = _region_contrast(out, mouth)  # 0..1
                m_str = mouth_strength * (0.6 + 0.9 * openness)  # cerrada ~0.6x, abierta ~1.5x
            out = apply_local_detail(out, mouth, amount=m_str * 1.4, radius=radius)
    return out


def unsharp(image: np.ndarray, amount: float = 0.6, radius: float = 0.0) -> np.ndarray:
    """Realce de nitidez por unsharp masking."""
    if amount <= 0:
        return image
    side = min(image.shape[:2])
    k = _odd(max(3, radius if radius > 1 else side * 0.03))
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    sharp = cv2.addWeighted(image.astype(np.float32), 1.0 + amount,
                            blurred.astype(np.float32), -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def apply_local_detail(
    face: np.ndarray, region_mask: np.ndarray, amount: float, radius: float = 0.0
) -> np.ndarray:
    """Aplica realce de detalle solo donde ``region_mask`` > 0, ponderado por su valor."""
    if amount <= 0 or region_mask.max() <= 0:
        return face
    detailed = unsharp(face, amount=amount, radius=radius)
    w = np.clip(region_mask, 0, 1)[:, :, None]
    out = detailed.astype(np.float32) * w + face.astype(np.float32) * (1.0 - w)
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_back_with_mask(
    frame: np.ndarray,
    face: np.ndarray,
    affine_matrix: np.ndarray,
    mask: np.ndarray,
    opacity: float = 1.0,
) -> np.ndarray:
    """Pega ``face`` en ``frame`` usando una ``mask`` (alineada) ya calculada."""
    h, w = frame.shape[:2]
    inverse = cv2.invertAffineTransform(affine_matrix)
    warped_face = cv2.warpAffine(face, inverse, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    warped_mask = cv2.warpAffine(mask.astype(np.float32), inverse, (w, h))
    warped_mask = np.clip(warped_mask * float(np.clip(opacity, 0.0, 1.0)), 0.0, 1.0)[:, :, None]
    out = warped_face * warped_mask + frame.astype(np.float32) * (1.0 - warped_mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_back(
    frame: np.ndarray,
    face: np.ndarray,
    affine_matrix: np.ndarray,
    mask_blur: float = 0.25,
    mask_padding: float = 0.0,
    opacity: float = 1.0,
) -> np.ndarray:
    """Paste-back con máscara rectangular suavizada (compatibilidad v1)."""
    fh, fw = face.shape[:2]
    mask = build_soft_mask(fh, fw, blur=mask_blur, padding=mask_padding)
    return paste_back_with_mask(frame, face, affine_matrix, mask, opacity)


def apply_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Transferencia de color Reinhard (LAB) de ``target`` a ``source``."""
    src = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean, src_std = src.mean(axis=(0, 1)), src.std(axis=(0, 1)) + 1e-6
    tgt_mean, tgt_std = tgt.mean(axis=(0, 1)), tgt.std(axis=(0, 1)) + 1e-6
    src = (src - src_mean) / src_std * tgt_std + tgt_mean
    src = np.clip(src, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src, cv2.COLOR_LAB2BGR)


def scale_affine(affine_matrix: np.ndarray, scale: float) -> np.ndarray:
    """Escala una matriz afín 2x3 para un recorte ``scale`` veces mayor."""
    return (affine_matrix.astype(np.float32) * float(scale)).astype(np.float32)


def limit_resolution(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Reduce un frame si su lado mayor supera ``max_side``. Devuelve (frame, factor)."""
    if max_side <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame, 1.0
    factor = max_side / float(longest)
    resized = cv2.resize(frame, (int(round(w * factor)), int(round(h * factor))),
                         interpolation=cv2.INTER_AREA)
    return resized, factor


def to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
