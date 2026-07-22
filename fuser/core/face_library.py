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

# Carpeta donde el Deep Swapper de FaceFusion escanea los .dfm entrenados por el
# usuario. create_static_model_set() los registra como model_id "custom/<slug>".
# (.assets/models es un junction a E: en esta máquina; escribir aquí es correcto.)
FF_CUSTOM_DFM_DIR = config.PROJECT_ROOT / "vendor" / "facefusion" / ".assets" / "models" / "custom"


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
    # borra también el .dfm asociado en la carpeta custom de FaceFusion
    man = _read_manifest(d) or {}
    if man.get("dfm"):
        dfm = FF_CUSTOM_DFM_DIR / (_slug(name) + ".dfm")
        try:
            dfm.unlink()
        except OSError:
            pass
    shutil.rmtree(d, ignore_errors=True)
    log.info("Cara '%s' borrada (%s)", name, d)
    return f"🗑️ Cara «{name}» borrada."


# --- Modelo entrenado (.dfm) por Cara -------------------------------------------
# Cada Cara puede tener un .dfm (DeepFaceLive) entrenado para esa identidad. El
# Deep Swapper de FaceFusion lo usa (geometría de cráneo completa) en vez del
# swapper one-shot. El .dfm se copia a la carpeta custom de FaceFusion y el
# model_id ("custom/<slug>") se guarda en manifest['dfm'].

def dfm_of(name: str):
    """model_id del .dfm de una Cara ("custom/<slug>") o None si no tiene."""
    d = face_dir(name)
    if not d.is_dir():
        return None
    man = _read_manifest(d) or {}
    model_id = man.get("dfm")
    if not model_id:
        return None
    # verificá que el archivo siga existiendo (si no, tratamos la Cara como one-shot)
    dfm = FF_CUSTOM_DFM_DIR / (_slug(name) + ".dfm")
    return model_id if dfm.is_file() else None


def has_dfm(name: str) -> bool:
    return dfm_of(name) is not None


def set_dfm(name: str, dfm_path: str) -> str:
    """Importa un .dfm entrenado y lo asocia a una Cara existente.

    Copia el archivo a la carpeta custom de FaceFusion como ``<slug>.dfm`` y
    escribe ``manifest['dfm'] = 'custom/<slug>'``. Al reiniciar Fuser, el Deep
    Swapper lo registra como ``custom/<slug>`` sin editar código.
    """
    name = (name or "").strip()
    d = face_dir(name)
    if not d.is_dir():
        raise ValueError(f"No existe la cara «{name}». Guardala primero con sus fotos.")
    if not dfm_path or Path(dfm_path).suffix.lower() != ".dfm" or not Path(dfm_path).is_file():
        raise ValueError("Subí un archivo .dfm válido (el modelo entrenado en DeepFaceLab).")
    size = Path(dfm_path).stat().st_size
    if size < 1_000_000:  # un .dfm real pesa decenas-cientos de MB
        raise ValueError("El .dfm parece truncado/corrupto (demasiado chico). Reintentá la copia.")

    slug = _slug(name)
    FF_CUSTOM_DFM_DIR.mkdir(parents=True, exist_ok=True)
    dst = FF_CUSTOM_DFM_DIR / f"{slug}.dfm"
    shutil.copyfile(dfm_path, dst)

    man = _read_manifest(d) or {"name": name, "slug": slug}
    man["dfm"] = f"custom/{slug}"
    man["dfm_size_mb"] = round(size / 1_048_576, 1)
    with open(d / MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)
    log.info("Cara '%s': .dfm importado (%s, %.1f MB) -> custom/%s", name, dst, size / 1_048_576, slug)
    return f"🧬 Modelo .dfm asociado a «{name}» ({man['dfm_size_mb']} MB). Reiniciá Fuser para usarlo."
