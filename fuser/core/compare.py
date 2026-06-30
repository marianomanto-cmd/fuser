"""Comparación A/B de modelos de swap one-shot — dentro de la app.

El swapper es la mayor palanca de calidad, y el mejor modelo depende de TU cara y
TU video (lo medimos: ghost_3 suele ganar en identidad, pero hififace/simswap pueden
ser mejores en caras concretas). Esta utilidad corre varios modelos sobre los MISMOS
frames clave del material del usuario y devuelve una galería para elegir a ojo.

⚠️ DirectML NO libera la VRAM entre cargas de modelos dentro de un mismo proceso (su
allocator la mantiene en pool), así que cargar varios swappers seguidos en el proceso
de Gradio cascaría. Por eso cada modelo se procesa en un SUBPROCESO aislado: al salir,
el SO libera su VRAM. Es más lento (recarga por modelo) pero robusto en 8 GB.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..config import Settings
from ..utils import image as imageutil
from ..utils import video as videoutil
from ..utils.logging import get_logger

log = get_logger(__name__)


def _largest_face_box(frame, analyser, pad: float = 0.7):
    faces = analyser.get_faces(frame)
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = f.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * (1 + pad)
    h, w = frame.shape[:2]
    return (int(max(0, cx - s / 2)), int(max(0, cy - s / 2)),
            int(min(w, cx + s / 2)), int(min(h, cy + s / 2)))


def _aligned_frames(video_path, idxs, res):
    """Devuelve ``[(idx, frame_limitada)]`` SOLO para los idx que decodifican OK.

    ``videoutil.get_frames_at`` descarta en silencio los frames que fallan al leer
    (p.ej. un keyframe cerca del final), devolviendo una lista MÁS CORTA sin decir
    cuál cayó. Si indexáramos por posición, el orquestador (recortes/etiquetas) y el
    worker (nombres de archivo por idx) se desalinearían o cascarían con IndexError.
    Pedimos idx por idx para conservar la correspondencia idx<->frame, y así ambos
    procesan EXACTAMENTE el mismo frame por idx.
    """
    pairs = []
    for idx in idxs:
        got = videoutil.get_frames_at(video_path, [idx])
        if got:
            pairs.append((idx, imageutil.limit_resolution(got[0], res)[0]))
    return pairs


def compare_models(source_paths, video_path, model_keys, base_settings: Settings,
                   n_frames: int = 3, progress=None):
    """Corre cada modelo (en subproceso) sobre ``n_frames`` frames clave del video.

    Devuelve ``[(rgb, caption)]`` agrupado por frame: ORIGINAL y cada modelo, recortado
    a la cara para juzgar identidad/detalle. Usa los ajustes de calidad actuales del
    usuario (enhancer, máscara, etc.) para que la comparación sea justa.
    """
    if not source_paths:
        raise ValueError("Subí al menos una imagen fuente (la cara a aplicar).")
    if not video_path:
        raise ValueError("Subí un video objetivo.")
    model_keys = [m for m in model_keys if m]
    if len(model_keys) < 2:
        raise ValueError("Elegí al menos 2 modelos para comparar.")

    res = base_settings.processing_resolution or 720
    info = videoutil.probe(video_path)
    idxs = videoutil.keyframe_indices(info.frame_count, n_frames)

    tmp = Path(tempfile.mkdtemp(prefix="fuser_cmp_"))
    sett = asdict(base_settings)
    sett["engine"] = config.ENGINE_FACEFUSION
    n = len(model_keys)
    for i, model in enumerate(model_keys):
        if progress:
            progress(i / (n + 0.2), f"Comparando {config.short_model(model)} "
                                    f"({i + 1}/{n}) · cada modelo en su proceso…")
        job = tmp / f"job_{i}.json"
        job.write_text(json.dumps({
            "model": model, "video": video_path, "sources": list(source_paths),
            "frames": int(n_frames), "res": int(res), "out": str(tmp), "settings": sett,
        }), encoding="utf-8")
        cmd = [sys.executable, "-m", "fuser.core.compare", "--job", str(job)]
        r = subprocess.run(cmd, cwd=str(config.PROJECT_ROOT),
                           capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("compare: modelo %s rc=%s · %s", model, r.returncode,
                        (r.stderr or "").strip()[-800:])

    if progress:
        progress(0.97, "Armando comparación…")

    # Recortes consistentes: una sola pasada de detección sobre los frames ORIGINALES
    # (ya terminaron los subprocesos, no hay contención de VRAM).
    from .memory_manager import MemoryManager
    from ..models.face_analyser import FaceAnalyser
    mm = MemoryManager(base_settings)
    det = FaceAnalyser(mm.analyser_providers(), mm.ctx_id(), mm.det_size)
    det.load()
    pairs = _aligned_frames(video_path, idxs, res)  # [(idx, orig)] alineado
    boxes = [_largest_face_box(f, det) for _, f in pairs]

    def crop(img, b):
        return img[b[1]:b[3], b[0]:b[2]] if (b and img is not None) else img

    items = []
    for k, (idx, orig) in enumerate(pairs):
        b = boxes[k]
        items.append((imageutil.to_rgb(crop(orig, b)), f"ORIGINAL · f{idx}"))
        for model in model_keys:
            p = tmp / f"{model}__f{idx:04d}.png"
            img = cv2.imread(str(p)) if p.exists() else None
            if img is not None:
                items.append((imageutil.to_rgb(crop(img, b)), f"{config.short_model(model)} · f{idx}"))
    return items


def _worker(job_path: str) -> None:
    """Subproceso: carga UN modelo, procesa los frames clave y guarda los resultados."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    from .pipeline import SwapPipeline

    s = Settings(**job["settings"])
    s.engine = config.ENGINE_FACEFUSION
    s.ff_swapper_model = job["model"]
    s.processing_resolution = int(job["res"])

    pipe = SwapPipeline(s)
    pipe.load_models(progress=lambda f, m="": None)
    srcs = [cv2.imread(p) for p in job["sources"]]
    srcs = [im for im in srcs if im is not None]
    pipe.prepare_source(srcs)

    info = videoutil.probe(job["video"])
    idxs = videoutil.keyframe_indices(info.frame_count, int(job["frames"]))
    out = Path(job["out"])
    # Mismo muestreo alineado que el orquestador → mismos frames por idx.
    for idx, work in _aligned_frames(job["video"], idxs, int(job["res"])):
        result = pipe.engine.process_frame(work, use_smoothing=False)
        cv2.imwrite(str(out / f"{job['model']}__f{idx:04d}.png"), result)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    _worker(ap.parse_args().job)
