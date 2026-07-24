"""Curación de facesets para entrenar un modelo `.dfm` (lógica compartida UI + CLI).

El parecido final del `.dfm` depende MÁS del faceset que de las horas de GPU. Acá
está la lógica que cura una carpeta/lista de imágenes de UNA persona y deja solo
las útiles, más un paquete `.zip` listo para el "extract faces" de DeepFaceLab.

Lo usan tanto ``scripts/prep_faceset.py`` (CLI) como la pestaña "🧬 Crear modelo"
de la UI. Procesa por RUTA; no recorta caras (eso lo hace DeepFaceLab) — copia las
imágenes BUENAS completas, renumeradas.
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .. import config
from ..utils.logging import get_logger

log = get_logger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_DET = None


def _detector():
    global _DET
    if _DET is None:
        from insightface.app import FaceAnalysis
        _DET = FaceAnalysis(name="buffalo_l", root=str(config.INSIGHTFACE_ROOT),
                            providers=["CPUExecutionProvider"])
        _DET.prepare(ctx_id=-1, det_size=(640, 640))
    return _DET


def iter_images(root: Path):
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def curate(
    image_paths: List,
    out_dir: Optional[Path] = None,
    min_face: int = 128,
    min_sharpness: float = 60.0,
    dedup: float = 0.96,
    copy: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Cura un faceset. Devuelve un reporte dict (ver claves abajo).

    Descarta ilegibles/sin-cara/varias-caras/cara-chica/borrosas/luz-mala,
    deduplica casi-idénticas, mide consistencia de identidad y cobertura de yaw.
    Si ``copy`` y ``out_dir``, copia las buenas renumeradas a ``out_dir``.
    """
    import cv2

    paths = [Path(p) for p in image_paths if p]
    det = _detector()
    kept, kept_embs, kept_yaw = [], [], []
    drop = Counter()
    dropped_examples: dict = {}
    total = max(1, len(paths))

    def _drop(reason, path):
        drop[reason] += 1
        dropped_examples.setdefault(reason, path.name)

    for i, p in enumerate(paths):
        if progress and (i % 5 == 0 or i == total - 1):
            progress(i / total, f"Analizando {i+1}/{total}…")
        img = cv2.imread(str(p))
        if img is None:
            _drop("ilegibles", p); continue
        faces = det.get(img)
        if not faces:
            _drop("sin_cara", p); continue
        if len(faces) > 1:
            _drop("varias_caras", p); continue
        f = faces[0]
        x1, y1, x2, y2 = f.bbox
        if min(x2 - x1, y2 - y1) < min_face:
            _drop("cara_chica", p); continue
        crop = img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if crop.size == 0:
            _drop("sin_cara", p); continue
        if cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < min_sharpness:
            _drop("borrosas", p); continue
        mean_v = float(crop.mean())
        if mean_v < 25 or mean_v > 235:
            _drop("luz_mala", p); continue
        emb = f.normed_embedding
        if kept_embs and float(np.max(np.dot(np.array(kept_embs), emb))) > dedup:
            _drop("casi_duplicadas", p); continue
        kps = f.kps
        io = float(np.linalg.norm(kps[0] - kps[1])) + 1e-6
        kept.append(p); kept_embs.append(emb)
        kept_yaw.append(float((kps[2][0] - (kps[0][0] + kps[1][0]) / 2) / io))

    report = {
        "scanned": len(paths),
        "kept": len(kept),
        "dropped": dict(drop),
        "dropped_examples": dropped_examples,
        "coverage": {"front": 0, "left": 0, "right": 0},
        "identity": {},
        "recommendations": [],
        "out_dir": None,
        "kept_paths": [str(p) for p in kept],
    }

    if kept:
        yy = np.array(kept_yaw)
        left = int((yy > 0.15).sum()); right = int((yy < -0.15).sum())
        report["coverage"] = {"front": len(yy) - left - right, "left": left, "right": right}
        centroid = np.mean(np.array(kept_embs), axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)
        cos = np.dot(np.array(kept_embs), centroid)
        outliers = [kept[i].name for i in range(len(kept)) if cos[i] < 0.30]
        report["identity"] = {"min_cos": round(float(cos.min()), 2),
                              "mean_cos": round(float(cos.mean()), 2),
                              "outliers": outliers}
        recs = report["recommendations"]
        if len(kept) < 300:
            recs.append(f"Tenés {len(kept)}; apuntá a 500-2000 para un .dfm decente. Sumá más fotos.")
        elif len(kept) < 500:
            recs.append(f"{len(kept)} es un piso; 500-2000 da mejor parecido.")
        if report["coverage"]["front"] < max(1, len(kept) // 6):
            recs.append("Faltan tomas FRONTALES.")
        if left < max(1, len(kept) // 8):
            recs.append("Faltan PERFILES hacia un lado.")
        if right < max(1, len(kept) // 8):
            recs.append("Faltan PERFILES hacia el otro lado.")
        if outliers:
            recs.append(f"⚠️ {len(outliers)} foto(s) parecen de OTRA persona: {', '.join(outliers[:5])}")

        if copy and out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(kept):
                shutil.copyfile(p, out_dir / f"{i:04d}{p.suffix.lower()}")
            report["out_dir"] = str(out_dir)
            if progress:
                progress(1.0, "Curado listo")

    return report


def make_bundle(curated_dir: Path, name: str) -> Path:
    """Empaqueta la carpeta curada en un .zip listo para subir a DeepFaceLab."""
    from ..core.face_library import _slug  # reutiliza el mismo slug
    slug = _slug(name) or "faceset"
    base = config.OUTPUTS_DIR / f"faceset_{slug}"
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = shutil.make_archive(str(base), "zip", str(curated_dir))
    return Path(zip_path)


def format_report_md(report: dict) -> str:
    """Reporte legible en markdown para la UI."""
    labels = {"ilegibles": "ilegibles", "sin_cara": "sin cara", "varias_caras": "varias caras",
              "cara_chica": "cara muy chica", "borrosas": "borrosas", "luz_mala": "muy oscuras/quemadas",
              "casi_duplicadas": "casi-duplicadas"}
    lines = [f"**{report['kept']} buenas** de {report['scanned']} escaneadas."]
    if report["dropped"]:
        drops = ", ".join(f"{labels.get(k, k)}: {v}" for k, v in report["dropped"].items())
        lines.append(f"Descartadas → {drops}.")
    c = report["coverage"]
    lines.append(f"Ángulos → frontal {c['front']} · perfil A {c['left']} · perfil B {c['right']}.")
    idn = report.get("identity") or {}
    if idn:
        lines.append(f"Identidad → cos min {idn.get('min_cos')} / media {idn.get('mean_cos')}.")
    for r in report.get("recommendations", []):
        lines.append(f"- {r}")
    return "\n\n".join(lines)
