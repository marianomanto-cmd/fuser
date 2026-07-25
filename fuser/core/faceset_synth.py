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

# Umbral: con menos de estas caras reales curadas se activa la síntesis.
# (Con videos de la PERSONA provistos, lo normal es superar esto de sobra y
# entrenar 100% con material real — la síntesis es solo relleno de emergencia.)
SYNTH_THRESHOLD = 150
# Cuántos crops sintéticos generar (env-tunable). Más = más cobertura, más horas.
def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(__import__("os").environ.get(name, default))))
    except (TypeError, ValueError):
        return default


SYNTH_TARGET = _env_int("FUSER_SYNTH_TARGET", 1000, 20, 5000)
# Peso de lo REAL cuando la síntesis se activa: PREDOMINA el material original
# (≈50/50 por duplicación; el resto de las veces el dataset es 100% real).
REAL_FRACTION = 0.5
MAX_DUP = 30


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
    """inswapper@512 + enhancer del preset PRO (pasada de textura de la cadena).

    Los valores de enhancer salen del PRESET EXPR_MAXID_CHAIN (una sola fuente de
    verdad: si se recalibra el preset medido, la síntesis lo hereda). Pixel boost
    512 (la receta MAX medida), NO el nativo 128: los donantes son crops de 512.
    """
    s = _settings_pass1()
    pro = config.EXPRESSION_PRESETS.get(config.EXPR_MAXID_CHAIN, {})
    for k in ("enhancer_model", "enhancer_blend", "codeformer_fidelity",
              "ff_enhancer_weight", "ff_swapper_weight"):
        if k in pro:
            setattr(s, k, pro[k])
    s.ff_swapper_model = "inswapper_128"; s.ff_pixel_boost = "512x512"
    s.ff_geometry_mask = False
    return s


def _run_pass_worker(pass_id: str, real_dir: Path, manifest: Path, out_dir: Path) -> int:
    """Cuerpo de UNA pasada (corre en su PROPIO proceso — ver _run_pass).

    Regla dura medida en esta máquina: DirectML retiene la VRAM entre modelos
    dentro de un mismo proceso; encadenar hififace→inswapper in-process ahoga la
    2ª carga (colgado sin error). Por eso cada pasada es un subproceso limpio.
    """
    import cv2
    from .pipeline import SwapPipeline

    settings = _settings_pass1() if pass_id == "pass1" else _settings_pass2()
    real_paths = [p for p in sorted(Path(real_dir).iterdir())
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")]
    real_images = [im for im in (cv2.imread(str(p)) for p in real_paths) if im is not None]
    in_files = [Path(l) for l in Path(manifest).read_text(encoding="utf-8").splitlines() if l.strip()]

    pipe = SwapPipeline(settings)
    pipe.load_models()
    pipe.prepare_source(real_images)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, f in enumerate(in_files):
        if i % 10 == 0:
            print(f"PASSPROG {i}/{len(in_files)}", flush=True)
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
        # Si FF no detectó cara, process_frame devuelve el frame INTACTO: dejarlo
        # pasar metería al DONANTE (otra persona) en el dataset de entrenamiento.
        import numpy as _np
        if out.shape == img.shape and float(_np.mean(_np.abs(
                out.astype(_np.int16) - img.astype(_np.int16)))) < 1.0:
            continue
        # PNG SIN PÉRDIDA: los intermedios no deben acumular recompresión JPEG
        # (síntesis en 2 pasadas + extract de DFL = 3 generaciones si fuera JPG).
        cv2.imwrite(str(out_dir / (f.stem + ".png")), out)
        done += 1
    print(f"PASSDONE {done}", flush=True)
    return done


def _run_pass(pass_id: str, real_dir: Path, in_files: List[Path], out_dir: Path,
              progress: Optional[Callable], frac0: float, frac1: float, label: str,
              timeout: int = 4 * 3600) -> int:
    """Lanza una pasada en SUBPROCESO propio y monitorea el avance por archivos."""
    import subprocess
    import sys
    import time as _t

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir.parent / (out_dir.name + "_manifest.txt")
    manifest.write_text("\n".join(str(f) for f in in_files), encoding="utf-8")
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(config.PROJECT_ROOT)
    env.setdefault("PYTHONIOENCODING", "utf-8"); env.setdefault("PYTHONUTF8", "1")
    logf = open(out_dir.parent / (out_dir.name + "_worker.log"), "ab")
    # como MÓDULO (-m): ejecutar el archivo directo rompería los imports relativos
    proc = subprocess.Popen(
        [sys.executable, "-m", "fuser.core.faceset_synth", pass_id,
         str(real_dir), str(manifest), str(out_dir)],
        cwd=str(config.PROJECT_ROOT), env=env,
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )
    total = max(1, len(in_files))
    t0 = _t.time()
    while proc.poll() is None:
        if _t.time() - t0 > timeout:
            proc.kill()
            raise RuntimeError(f"síntesis {pass_id}: timeout tras {timeout}s")
        n = len(list(out_dir.glob("*.png")))
        if progress:
            progress(frac0 + (frac1 - frac0) * n / total, f"{label} {n}/{total}…")
        _t.sleep(10)
    logf.close()
    done = len(list(out_dir.glob("*.png")))
    if proc.returncode != 0 and done == 0:
        raise RuntimeError(f"síntesis {pass_id} falló (rc={proc.returncode}); "
                           f"log: {out_dir.parent / (out_dir.name + '_worker.log')}")
    return done


def synthesize(real_dir: Path, donor_files: List[Path], out_dir: Path,
               target: int = SYNTH_TARGET,
               progress: Optional[Callable] = None) -> dict:
    """Genera el faceset sintético en ``out_dir`` (crops JPG listos para extract).

    ``real_dir``: fotos reales curadas. ``donor_files``: crops de caras donantes
    (alineados; poses/luces variadas). Devuelve {'synthetic': n, 'donors_used': m}.
    """
    real_dir = Path(real_dir)
    if not any(real_dir.iterdir()):
        raise ValueError("No pude leer las fotos reales para la síntesis.")

    donors = list(donor_files)
    random.Random(7).shuffle(donors)   # determinista: mismo sample entre corridas
    donors = donors[:target]
    if not donors:
        raise ValueError("No hay caras donantes para sintetizar (faceset genérico/videos).")

    tmp = out_dir.parent / (out_dir.name + "_p1")
    n1 = _run_pass("pass1", real_dir, donors, tmp,
                   progress, 0.0, 0.55, "Síntesis 1/2 · forma (hififace)")
    if n1 == 0:
        raise RuntimeError("La síntesis no produjo caras en la pasada de forma.")
    mid_files = sorted(tmp.glob("*.png"))
    n2 = _run_pass("pass2", real_dir, mid_files, out_dir,
                   progress, 0.55, 0.9, "Síntesis 2/2 · textura (inswapper)")
    # PUERTA DE CALIDAD: los donantes difíciles (oclusiones fuertes: gorras,
    # lentes, oscuridad) salen mal transformados — medido en stock: crops con la
    # cara del DONANTE casi intacta. Cada crop se mide contra la persona real y
    # los flojos NO entran al dataset (mejor menos y bueno que mucho y sucio).
    n2 = _identity_gate(real_dir, out_dir, progress)
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


# Umbral de la puerta de identidad (cos ArcFace del crop sintético vs persona).
SYNTH_MIN_ID = float(__import__("os").environ.get("FUSER_SYNTH_MIN_ID", "0.35"))


def _identity_gate(real_dir: Path, out_dir: Path,
                   progress: Optional[Callable] = None) -> int:
    """Borra los crops sintéticos que no llevan la identidad de la persona.

    CPU (buffalo_l): centroide de las fotos reales → cos de cada crop; borra los
    de cos < SYNTH_MIN_ID o sin cara detectable. Devuelve cuántos quedaron.
    """
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    det = FaceAnalysis(name="buffalo_l", root=str(config.INSIGHTFACE_ROOT),
                       providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=-1, det_size=(640, 640))

    def _emb(img):
        fs = det.get(img)
        if not fs:
            # Los crops alineados 512 llenan el frame con la cara y el detector
            # (entrenado para caras chicas) suele NO verla: reintentar con lienzo
            # acolchado 2x (la cara pasa a ocupar ~50% y detecta bien).
            h, w = img.shape[:2]
            canvas = np.full((h * 2, w * 2, 3), 127, np.uint8)
            y0, x0 = h // 2, w // 2
            canvas[y0:y0 + h, x0:x0 + w] = img
            fs = det.get(canvas)
            if not fs:
                return None
        b = max(fs, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        return b.normed_embedding

    reals = []
    for p in sorted(Path(real_dir).iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            img = cv2.imread(str(p))
            e = _emb(img) if img is not None else None
            if e is not None:
                reals.append(e)
    if not reals:
        log.warning("identity_gate: sin embeddings reales; no filtro.")
        return len(list(out_dir.glob("*.png")))
    cent = np.mean(reals, axis=0)
    cent /= np.linalg.norm(cent) + 1e-8

    files = sorted(out_dir.glob("*.png"))
    kept = dropped = 0
    for i, f in enumerate(files):
        if progress and i % 15 == 0:
            progress(0.9 + 0.1 * i / max(1, len(files)),
                     f"Puerta de calidad {i}/{len(files)}…")
        img = cv2.imread(str(f))
        e = _emb(img) if img is not None else None
        score = float(np.dot(e, cent)) if e is not None else -1.0
        if score < SYNTH_MIN_ID:
            f.unlink(missing_ok=True)
            dropped += 1
        else:
            kept += 1
    log.info("identity_gate: %d ok · %d descartados (cos<%s o sin cara).",
             kept, dropped, SYNTH_MIN_ID)
    return kept


def real_duplication(n_real: int, n_synth: int) -> int:
    """Cuántas copias de cada foto real anclan ~REAL_FRACTION del dataset."""
    if n_real <= 0:
        return 0
    dup = int(round((REAL_FRACTION * n_synth) / max(1, n_real * (1 - REAL_FRACTION))))
    return max(1, min(MAX_DUP, dup))


if __name__ == "__main__":  # worker de una pasada (subproceso; ver _run_pass)
    import sys as _sys
    _sys.exit(0 if _run_pass_worker(_sys.argv[1], Path(_sys.argv[2]),
                                    Path(_sys.argv[3]), Path(_sys.argv[4])) >= 0 else 1)
