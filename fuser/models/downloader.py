"""Descarga perezosa y verificable de modelos ONNX.

Diseño tolerante a fallos:
- Si el archivo ya existe localmente, no se descarga.
- Se intentan varias URLs (espejos) en orden.
- Descarga atómica (a ``.part`` y luego ``rename``) para no dejar archivos corruptos.
- Verificación opcional de SHA-256.
- Si todo falla, se lanza un error con instrucciones claras de descarga manual.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ..config import MODEL_REGISTRY, MODELS_DIR, ModelInfo, ensure_dirs
from ..utils.logging import get_logger

log = get_logger(__name__)

ProgressCb = Optional[Callable[[float, str], None]]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_one(url: str, dest: Path, progress: ProgressCb = None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fuser/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(tmp, "wb") as out:
            while True:
                buf = resp.read(1 << 20)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                if progress and total:
                    progress(downloaded / total, f"Descargando {dest.name}")
    tmp.replace(dest)


def ensure_model(key: str, progress: ProgressCb = None) -> Path:
    """Garantiza que el modelo ``key`` está en disco; lo descarga si hace falta.

    Devuelve la ruta local. Lanza ``ModelDownloadError`` con instrucciones si no
    se puede obtener por ninguna vía.
    """
    ensure_dirs()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Modelo desconocido: {key}")
    info: ModelInfo = MODEL_REGISTRY[key]
    dest = info.path

    if dest.exists() and dest.stat().st_size > 0:
        if info.sha256 and _sha256(dest) != info.sha256:
            log.warning("Hash incorrecto en %s; se vuelve a descargar.", dest.name)
            dest.unlink(missing_ok=True)
        else:
            return dest

    errors = []
    for url in info.urls:
        try:
            log.info("Descargando %s desde %s", info.filename, url)
            _download_one(url, dest, progress)
            if info.sha256 and _sha256(dest) != info.sha256:
                dest.unlink(missing_ok=True)
                raise RuntimeError("checksum no coincide")
            log.info("Modelo listo: %s", dest)
            return dest
        except Exception as exc:  # pragma: no cover - depende de la red
            log.warning("Fallo al descargar desde %s: %s", url, exc)
            errors.append(f"  - {url}\n      {exc}")

    raise ModelDownloadError(info, errors)


class ModelDownloadError(RuntimeError):
    """Error de descarga con mensaje accionable para el usuario."""

    def __init__(self, info: ModelInfo, errors):
        msg = (
            f"\n❌ No se pudo descargar el modelo '{info.key}' ({info.filename}).\n\n"
            f"Intentos:\n" + "\n".join(errors) + "\n\n"
            f"➡️  Solución manual: descarga '{info.filename}' desde una de las URLs de "
            f"arriba (o búscalo en Hugging Face / facefusion-assets) y colócalo en:\n"
            f"     {info.path}\n"
        )
        super().__init__(msg)
        self.info = info


def is_downloaded(key: str) -> bool:
    info = MODEL_REGISTRY.get(key)
    return bool(info and info.path.exists() and info.path.stat().st_size > 0)
