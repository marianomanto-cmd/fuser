"""Síntesis de faceset: de ~10 fotos reales a un dataset de entrenamiento completo.

Problema: DeepFaceLab necesita cientos-miles de ejemplos variados de la persona;
con ~10 fotos el SAEHD se sobreajusta y el .dfm sale pobre. Solución (técnica de
"bootstrapping de identidad"): usar NUESTRO motor one-shot de máxima identidad
para estampar la cara de las fotos reales sobre miles de caras DONANTES en todas
las poses/expresiones/luces, y entrenar con ese set expandido.

Pipeline (la misma cadena forma→textura del modo 🎯➕ PRO, medida como la de
mayor parecido, aplicada crop a crop):
  pasada 1: hififace + máscara de geometría (impone nariz/cráneo de la foto)
  pasada 2: inswapper + CodeFormer fiel (reinyecta textura/nitidez de identidad)
Las fotos REALES se duplican para anclar el entrenamiento (~15% del dataset):
lo real manda, lo sintético da cobertura de poses.

Techo honesto: el .dfm resultante hereda la identidad que logra el one-shot
(estabilizada y a cara completa, que es la ganancia del entrenamiento). Más
fotos reales variadas siguen siendo mejores que la síntesis.
"""
from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Callable, List, Optional

from .. import config
from ..utils.logging import get_logger

log = get_logger(__name__)

# Umbral: con menos de estas fotos reales curadas se activa la síntesis.
SYNTH_THRESHOLD = 150
# Cuántos crops sintéticos generar (env-tunable). Más = más cobertura, más horas.
SYNTH_TARGET = int(__import__("os").environ.get("FUSER_SYNTH_TARGET", "1000"))
# Peso de lo real en el dataset final (fracción aproximada, vía duplicación).
REAL_FRACTION = 0.15
MAX_DUP = 50


def _settings_pass1() -> config.Settings:
    """hififace + geometría (como la pasada 1 de la cadena PRO), sin extras."""
    s = config.Settings()
    for k, v in config.EXPRESSION_PRESETS[config.EXPR_MAXIDENTITY].items():
        setattr(s, k, v)
    s.expression_mode = config.EXPR_MAXIDENTITY
    s.enhancer_model = "none"; s.enhancer_blend = 0.0
    s.temporal_smoothing = False; s.two_pass_temporal = False; s.qc_second_pass = False
    s.skin_detail = 0.0; s.eye_preservation = 0.0
    s.color_harmonize = False; s.chain_shape_then_texture = False
    return s


def _settings_pass2() -> config.Settings:
    """inswapper@512 + CodeFormer fiel (pasada de textura de la cadena PRO).

    Pixel boost 512 (la receta MAX medida), NO el nativo 128: los donantes son
    crops de 512 px y bajar el swap a 128 tiraría detalle.
    """
    s = _settings_pass1()
    s.ff_swapper_model = "inswapper_128"; s.ff_pixel_boost = "512x512"
    s.ff_geometry_mask = False
    s.enhancer_model = "codeformer"; s.enhancer_blend = 0.7
    s.codeformer_fidelity = 0.5; s.ff_enhancer_weight = 0.7
    return s


def _run_pass(settings, real_images, in_files: List[Path], out_dir: Path,
              progress: Optional[Callable], frac0: float, frac1: float, label: str) -> int:
    """Corre una pasada de swap sobre una lista de crops. Devuelve nº generados."""
    import cv2
    from .pipeline import SwapPipeline

    pipe = SwapPipeline(settings)
    pipe.load_models()
    pipe.prepare_source(real_images)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    total = max(1, len(in_files))
    for i, f in enumerate(in_files):
        if progress and i % 20 == 0:
            progress(frac0 + (frac1 - frac0) * i / total, f"{label} {i}/{total}…")
        img = cv2.imread(str(f))
        if img is None:
            continue
        try:
            out = pipe.engine.process_frame(img, use_smoothing=False)
        except Exception as exc:  # cara no detectada u otro fallo puntual: saltar
            log.debug("synth: fallo en %s (%s)", f.name, exc)
            continue
        if out is None:
            continue
        # PNG SIN PÉRDIDA: los intermedios no deben acumular recompresión JPEG
        # (síntesis en 2 pasadas + extract de DFL = 3 generaciones si fuera JPG).
        cv2.imwrite(str(out_dir / (f.stem + ".png")), out)
        done += 1
    # liberar el modelo/pool antes de la siguiente pasada (misma process; DirectML)
    try:
        pipe.engine.unload()
    except Exception:
        pass
    del pipe
    gc.collect()
    return done


def synthesize(real_dir: Path, donor_files: List[Path], out_dir: Path,
               target: int = SYNTH_TARGET,
               progress: Optional[Callable] = None) -> dict:
    """Genera el faceset sintético en ``out_dir`` (crops JPG listos para extract).

    ``real_dir``: fotos reales curadas. ``donor_files``: crops de caras donantes
    (alineados; poses/luces variadas). Devuelve {'synthetic': n, 'donors_used': m}.
    """
    import cv2

    real_paths = [p for p in sorted(Path(real_dir).iterdir())
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")]
    real_images = [im for im in (cv2.imread(str(p)) for p in real_paths) if im is not None]
    if not real_images:
        raise ValueError("No pude leer las fotos reales para la síntesis.")

    donors = list(donor_files)
    random.Random(7).shuffle(donors)   # determinista: mismo sample entre corridas
    donors = donors[:target]
    if not donors:
        raise ValueError("No hay caras donantes para sintetizar (faceset genérico/videos).")

    tmp = out_dir.parent / (out_dir.name + "_p1")
    n1 = _run_pass(_settings_pass1(), real_images, donors, tmp,
                   progress, 0.0, 0.55, "Síntesis 1/2 · forma (hififace)")
    if n1 == 0:
        raise RuntimeError("La síntesis no produjo caras en la pasada de forma.")
    mid_files = sorted(tmp.glob("*.png"))
    n2 = _run_pass(_settings_pass2(), real_images, mid_files, out_dir,
                   progress, 0.55, 1.0, "Síntesis 2/2 · textura (inswapper)")
    # limpieza de la etapa intermedia
    for f in mid_files:
        try:
            f.unlink()
        except OSError:
            pass
    try:
        tmp.rmdir()
    except OSError:
        pass
    if n2 == 0:
        raise RuntimeError("La síntesis no produjo caras en la pasada de textura.")
    log.info("Faceset sintético: %d crops (de %d donantes).", n2, len(donors))
    return {"synthetic": n2, "donors_used": len(donors)}


def real_duplication(n_real: int, n_synth: int) -> int:
    """Cuántas copias de cada foto real anclan ~REAL_FRACTION del dataset."""
    if n_real <= 0:
        return 0
    dup = int(round((REAL_FRACTION * n_synth) / max(1, n_real * (1 - REAL_FRACTION))))
    return max(1, min(MAX_DUP, dup))
