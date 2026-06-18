"""Auto-instalación de FaceFusion (sin pasos manuales).

El usuario no tiene que clonar nada: FaceFusion se descarga e instala solo,
una única vez, en ``vendor/facefusion`` dentro del proyecto. Esto lo dispara:

- ``scripts/setup.sh`` / ``setup.bat`` (durante la instalación), o
- el propio motor al **seleccionar "FaceFusion (Alta Calidad)"** en la UI
  (auto-bootstrap), si ``settings.ff_auto_install`` está activo (por defecto).

Tras instalar las dependencias de FaceFusion se **restauran los pines de Fuser**
(gradio 5, numpy<2) por si su instalación los cambia.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from ..config import PROJECT_ROOT
from ..utils.logging import get_logger

log = get_logger(__name__)

# Repositorio y versión (anclada para que el adaptador 3.x encaje). Ambos
# configurables por variable de entorno.
FACEFUSION_REPO = os.environ.get("FUSER_FACEFUSION_REPO", "https://github.com/facefusion/facefusion")
FACEFUSION_VERSION = os.environ.get("FUSER_FACEFUSION_VERSION", "3.1.1")

ProgressCb = Optional[Callable[[float, str], None]]


class FaceFusionNotAvailable(RuntimeError):
    def __init__(self, detail: str = ""):
        super().__init__(
            "FaceFusion no está disponible.\n"
            "Debería instalarse solo al elegir el motor o con:\n"
            "    python scripts/install_facefusion.py\n"
            f"{('Detalle: ' + detail) if detail else ''}\n"
            "Mientras tanto puedes usar el motor 'InsightFace (Rápido)'."
        )


def vendor_dir() -> Path:
    return PROJECT_ROOT / "vendor" / "facefusion"


def ensure_on_path() -> None:
    """Añade ``vendor/facefusion`` a sys.path si existe (para poder importarlo)."""
    cand = vendor_dir()
    if cand.exists():
        p = str(cand)
        if p not in sys.path:
            sys.path.insert(0, p)


def is_available() -> bool:
    """True si ``import facefusion`` funciona (tras añadir el vendor al path)."""
    ensure_on_path()
    import importlib

    try:
        importlib.import_module("facefusion")
        return True
    except Exception:
        return False


def _run(cmd: list, **kw) -> int:
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, **kw).returncode


def _clone(progress: ProgressCb = None) -> None:
    target = vendor_dir()
    if (target / "facefusion").exists() or (target / ".git").exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.15, "Descargando FaceFusion…")
    # Intenta la versión anclada; si el tag no existe, usa la rama por defecto.
    tagged = _run(["git", "clone", "--depth", "1", "--branch", FACEFUSION_VERSION,
                   FACEFUSION_REPO, str(target)])
    if tagged != 0:
        log.warning("No se pudo clonar el tag %s; usando rama por defecto.", FACEFUSION_VERSION)
        if _run(["git", "clone", "--depth", "1", FACEFUSION_REPO, str(target)]) != 0:
            raise FaceFusionNotAvailable("no se pudo clonar el repositorio (¿git instalado? ¿red?)")


def _pip_install_requirements(progress: ProgressCb = None) -> None:
    req = vendor_dir() / "requirements.txt"
    if req.exists():
        if progress:
            progress(0.5, "Instalando dependencias de FaceFusion…")
        _run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    else:
        log.warning("FaceFusion no trae requirements.txt; puede requerir 'python install.py'.")


def _restore_fuser_pins(progress: ProgressCb = None) -> None:
    if progress:
        progress(0.85, "Restaurando dependencias de Fuser…")
    _run([sys.executable, "-m", "pip", "install", "-U", "gradio>=5,<6", "numpy<2"])


def install(progress: ProgressCb = None) -> bool:
    """Clona + instala FaceFusion en ``vendor/`` y verifica que importa."""
    log.info("Instalando FaceFusion (una sola vez) en %s", vendor_dir())
    _clone(progress)
    _pip_install_requirements(progress)
    _restore_fuser_pins(progress)
    ensure_on_path()
    if not is_available():
        raise FaceFusionNotAvailable(
            "instalación completada pero 'import facefusion' falla. "
            "Quizá necesites su instalador propio: "
            f"cd {vendor_dir()} && python install.py --onnxruntime cuda"
        )
    if progress:
        progress(1.0, "FaceFusion listo.")
    log.info("FaceFusion instalado correctamente.")
    return True


def ensure(progress: ProgressCb = None, auto: bool = True) -> None:
    """Garantiza FaceFusion disponible; lo instala automáticamente si ``auto``."""
    if is_available():
        return
    if not auto:
        raise FaceFusionNotAvailable("auto-instalación desactivada (ff_auto_install=False)")
    install(progress)
