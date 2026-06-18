"""Pre-descarga de modelos (opcional).

Permite bajar todos los modelos por adelantado para que el primer uso de la app
sea instantáneo. Útil también para construir imágenes/Spaces con caché.

Uso:
    python scripts/download_models.py                 # descarga los recomendados
    python scripts/download_models.py --all           # descarga todos
    python scripts/download_models.py --only inswapper_128 gfpgan_1.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuser import config  # noqa: E402
from fuser.models.downloader import ensure_model  # noqa: E402

RECOMMENDED = ["inswapper_128", "gfpgan_1.4"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga de modelos de Fuser")
    parser.add_argument("--all", action="store_true", help="Descargar todos los modelos")
    parser.add_argument("--only", nargs="*", help="Descargar solo estas claves de modelo")
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

    if failures:
        print(f"\n⚠️  Fallaron: {', '.join(failures)}. Revisa las URLs o descárgalos a mano "
              f"en {config.MODELS_DIR}.")
        return 1
    print("\n✅ Todos los modelos están listos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
