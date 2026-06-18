"""Pre-descarga de modelos.

Baja por adelantado los modelos para que el primer uso sea instantáneo. Lo
ejecuta ``scripts/setup.sh``/``setup.bat`` durante la instalación, así la app
"se pone a bajar sola" lo que falte (los `.onnx` de Fuser y el detector
buffalo_l de InsightFace). FaceFusion es aparte (motor opcional).

Uso:
    python scripts/download_models.py                 # recomendados + detector
    python scripts/download_models.py --all           # todos los modelos
    python scripts/download_models.py --only inswapper_128 gfpgan_1.4
    python scripts/download_models.py --no-detector   # sin bajar buffalo_l
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuser import config  # noqa: E402
from fuser.models.downloader import ensure_model  # noqa: E402

# inswapper (swap) + gfpgan/codeformer (enhancers; codeformer lo usa el modo
# musical y el realce localizado de boca).
RECOMMENDED = ["inswapper_128", "gfpgan_1.4", "codeformer"]


def _download_detector() -> bool:
    """Pre-descarga el pack buffalo_l de InsightFace (carpeta `insightface`)."""
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", root=str(config.INSIGHTFACE_ROOT))
        app.prepare(ctx_id=-1, det_size=(320, 320))  # ctx_id=-1 = CPU, solo para bajar
        return True
    except Exception as exc:  # pragma: no cover - depende del entorno
        print(f"  ⚠️  buffalo_l no se pudo pre-bajar ({exc}); se bajará en el primer uso.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga de modelos de Fuser")
    parser.add_argument("--all", action="store_true", help="Descargar todos los modelos")
    parser.add_argument("--only", nargs="*", help="Descargar solo estas claves de modelo")
    parser.add_argument("--no-detector", action="store_true", help="No pre-bajar buffalo_l")
    args = parser.parse_args()

    if args.only:
        keys = args.only
    elif args.all:
        keys = list(config.MODEL_REGISTRY.keys())
    else:
        keys = RECOMMENDED

    print(f"Modelos a descargar: {', '.join(keys)}\n")
    failures = []
    for key in keys:
        try:
            path = ensure_model(key, progress=lambda f, m="": None)
            print(f"  ✅ {key:28s} -> {path}")
        except Exception as exc:
            print(f"  ❌ {key:28s} -> {exc}")
            failures.append(key)

    if not (args.only or args.no_detector):
        print("\nDetector InsightFace (buffalo_l):")
        if _download_detector():
            print(f"  ✅ buffalo_l -> {config.INSIGHTFACE_ROOT}/models/buffalo_l")

    if failures:
        print(f"\n⚠️  Fallaron: {', '.join(failures)}. Revisa las URLs o descárgalos a mano "
              f"en {config.MODELS_DIR}.")
        return 1
    print("\n✅ Modelos listos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
