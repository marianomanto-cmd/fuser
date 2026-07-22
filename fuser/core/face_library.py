"""Biblioteca de Caras: identidades fuente guardadas (multi-referencia).

Cada "Cara" (p.ej. "Cara 1") es una carpeta bajo ``config.FACES_DIR`` con las
fotos de esa persona + un ``manifest.json``. Al elegir una Cara en la UI, el
pipeline usa TODAS sus fotos como fuente MULTI-REFERENCIA (más ángulos y
expresiones = más identidad) en vez de subir fotos cada vez.

No entrena nada: NO es un ``.dfm``; reusa el motor one-shot (inswapper/hififace).
Queda diseñado para que, en el futuro, cada Cara pueda además guardar un ``.dfm``
entrenado por identidad (campo ``dfm`` en el manifest) sin cambiar este API ni el
flujo de la UI.

Todo el manejo es por RUTAS de archivo (copiar/borrar): este módulo nunca abre
ni "mira" el contenido de las imágenes del usuario.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional

from .. import config
from ..utils.logging import get_logger

log = get_logger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST = "manifest.json"
MAX_IMAGES = 40  # tope sano de fotos por cara (multi-ref satura mucho antes)


def _slug(name: str) -> str:
    """Nombre de carpeta seguro y estable a partir del nombre visible."""
    s = re.sub(r"[^\w\- ]+", "", (name or "").strip(), flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s).strip("_.")
    return s.lower()[:64]


def faces_root() -> Path:
    config.FACES_DIR.mkdir(parents=True, exist_ok=True)
    return config.FACES_DIR


def face_dir(name: str) -> Path:
    return faces_root() / _slug(name)


def _read_manifest(d: Path) -> Optional[dict]:
    try:
        with open(d / MANIFEST, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def list_faces() -> List[str]:
    """Nombres visibles de las caras guardadas (orden alfabético)."""
    root = faces_root()
    names = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        man = _read_manifest(d)
        if man and man.get("name") and face_images(man["name"]):
            names.append(man["name"])
    return sorted(names, key=lambda s: s.lower())


def face_images(name: str) -> List[str]:
    """Rutas de las fotos guardadas de una cara (orden estable)."""
    d = face_dir(name)
    if not d.is_dir():
        return []
    return sorted(
        str(p) for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def save_face(name: str, image_paths: List[str]) -> str:
    """Crea/reemplaza una cara con las fotos dadas. Devuelve un resumen.

    Copia las imágenes a ``FACES_DIR/<slug>/`` y escribe el manifest. Reemplaza
    por completo las fotos previas de esa cara (guardar = estado final deseado).
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Poné un nombre para la cara (p.ej. 'Cara 1').")
    valid = [p for p in (image_paths or []) if p and Path(p).suffix.lower() in IMG_EXTS and Path(p).is_file()]
    if not valid:
        raise ValueError("Subí al menos una foto (jpg/png/webp) de esta persona.")
    valid = valid[:MAX_IMAGES]

    d = face_dir(name)
    # limpia fotos previas (reemplazo total), conserva la carpeta
    if d.exists():
        for p in d.iterdir():
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
    d.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, src in enumerate(valid):
        ext = Path(src).suffix.lower()
        dst = d / f"img_{i:02d}{ext}"
        try:
            shutil.copyfile(src, dst)
            saved += 1
        except OSError as exc:
            log.warning("No se pudo copiar %s: %s", src, exc)
    if saved == 0:
        raise ValueError("No se pudo guardar ninguna foto (¿permisos/formato?).")

    manifest = {
        "name": name,
        "slug": _slug(name),
        "count": saved,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dfm": None,  # reservado: futuro modelo .dfm entrenado por identidad
    }
    with open(d / MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log.info("Cara '%s' guardada con %d foto(s) en %s", name, saved, d)
    return f"✅ Cara «{name}» guardada con {saved} foto(s)."


def delete_face(name: str) -> str:
    d = face_dir(name)
    if not d.is_dir():
        raise ValueError(f"No existe la cara «{name}».")
    shutil.rmtree(d, ignore_errors=True)
    log.info("Cara '%s' borrada (%s)", name, d)
    return f"🗑️ Cara «{name}» borrada."
