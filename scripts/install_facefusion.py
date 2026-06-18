"""Instala el motor opcional FaceFusion (cross-platform).

    python scripts/install_facefusion.py

Clona FaceFusion en vendor/facefusion, instala sus dependencias en el entorno
actual y restaura los pines de Fuser. Lo usan también ``setup.sh`` / ``setup.bat``
y el auto-bootstrap de la app (al elegir el motor en la UI).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuser.engines.facefusion_bootstrap import FACEFUSION_VERSION, install, is_available  # noqa: E402


def main() -> int:
    if is_available():
        print("✅ FaceFusion ya está disponible.")
        return 0
    print(f"➡️  Instalando FaceFusion ({FACEFUSION_VERSION}) — esto puede tardar unos minutos…")
    try:
        install(progress=lambda f, m="": print(f"   [{int(f*100):3d}%] {m}"))
    except Exception as exc:
        print(f"\n❌ {exc}")
        return 1
    print("\n✅ FaceFusion instalado. En la UI elige 'FaceFusion (Alta Calidad)'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
