"""Catálogo de modelos del módulo *Imagen → Vídeo* y utilidades de verificación.

Dos formas de comprobar que los modelos están listos:

1. **Remota (sin acceso al disco de ComfyUI):** a partir de ``/object_info`` de
   ComfyUI sabemos qué ficheros ve cada cargador (``UnetLoaderGGUF``,
   ``VAELoader``, ``CheckpointLoaderSimple``…). Es lo que usa la UI.
2. **Local (descarga):** si tienes la ruta a la instalación de ComfyUI, podemos
   descargar los modelos a sus carpetas (``models/unet``, ``models/vae``…).
   Lo usa ``scripts/setup_i2v.py``.

Los pesos NO se versionan en este repo (igual que el resto de modelos de Fuser).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..utils.logging import get_logger

log = get_logger(__name__)

ProgressCb = Optional[Callable[[float, str], None]]


@dataclass(frozen=True)
class I2VModelFile:
    """Un fichero de modelo que ComfyUI necesita para esta función."""

    key: str
    filename: str            # nombre con el que ComfyUI lo ve (y nombre local)
    repo: str                # repo de Hugging Face "owner/name"
    path_in_repo: str        # ruta del fichero dentro del repo
    subfolder: str           # subcarpeta de ComfyUI/models/ donde va
    size_gb: float
    kind: str                # "video" | "audio"
    gated: bool = False      # requiere aceptar licencia + token de HF
    note: str = ""

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.path_in_repo}"


# --- Wan 2.2 vídeo — modelos instalados en esta máquina ----------------------
# Por defecto el modo RÁPIDO usa el TI2V-5B (cabe entero en 8 GB). El 14B (dual
# experto, Q3_K_S) es la opción de máxima calidad pero lenta (offload a RAM).
WAN_I2V_MODELS: List[I2VModelFile] = [
    I2VModelFile(
        key="wan_ti2v_5b",
        filename="Wan2.2-TI2V-5B-Q4_K_M.gguf",
        repo="QuantStack/Wan2.2-TI2V-5B-GGUF",
        path_in_repo="Wan2.2-TI2V-5B-Q4_K_M.gguf",
        subfolder="unet",
        size_gb=3.4,
        kind="video",
        note="Modelo 5B (rápido, cabe en 8 GB sin offload). Usa el VAE 2.2.",
    ),
    I2VModelFile(
        key="wan_high_q3",
        filename="Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf",
        repo="QuantStack/Wan2.2-I2V-A14B-GGUF",
        path_in_repo="HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf",
        subfolder="unet",
        size_gb=6.5,
        kind="video",
        note="Experto de ALTO ruido (14B, máxima calidad, lento en 8 GB).",
    ),
    I2VModelFile(
        key="wan_low_q3",
        filename="Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf",
        repo="QuantStack/Wan2.2-I2V-A14B-GGUF",
        path_in_repo="LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf",
        subfolder="unet",
        size_gb=6.5,
        kind="video",
        note="Experto de BAJO ruido (14B).",
    ),
    I2VModelFile(
        key="umt5_fp8",
        filename="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        repo="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        path_in_repo="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        subfolder="text_encoders",
        size_gb=5.0,
        kind="video",
        note="Codificador de texto UMT5-XXL (fp8).",
    ),
    I2VModelFile(
        key="wan22_vae",
        filename="wan2.2_vae.safetensors",
        repo="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        path_in_repo="split_files/vae/wan2.2_vae.safetensors",
        subfolder="vae",
        size_gb=0.25,
        kind="video",
        note="VAE de Wan 2.2 (lo usa el TI2V-5B).",
    ),
    I2VModelFile(
        key="wan_vae",
        filename="wan_2.1_vae.safetensors",
        repo="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        path_in_repo="split_files/vae/wan_2.1_vae.safetensors",
        subfolder="vae",
        size_gb=0.25,
        kind="video",
        note="VAE de Wan 2.1 (lo usa el 14B I2V).",
    ),
]

# --- Stable Audio Open (audio) ----------------------------------------------
AUDIO_MODELS: List[I2VModelFile] = [
    I2VModelFile(
        key="stable_audio",
        filename="stable_audio_open_1.0.safetensors",
        # Copia REPACKAGED de Comfy-Org (misma org que los modelos de Wan): es el
        # modelo BASE, NO-gated, así que se descarga sin token (el repo oficial de
        # stabilityai está gated). El fichero ya viene con ese nombre en checkpoints/.
        repo="Comfy-Org/stable-audio-open-1.0_repackaged",
        path_in_repo="stable-audio-open-1.0.safetensors",
        subfolder="checkpoints",
        size_gb=4.85,
        kind="audio",
        note="Stable Audio Open 1.0 (checkpoint all-in-one: modelo + T5 + VAE).",
    ),
    I2VModelFile(
        key="t5_base",
        filename="t5_base.safetensors",
        repo="google-t5/t5-base",
        path_in_repo="model.safetensors",
        subfolder="text_encoders",
        size_gb=0.9,
        kind="audio",
        note="Codificador de texto de Stable Audio. Renómbralo a t5_base.safetensors.",
    ),
]

ALL_MODELS: List[I2VModelFile] = WAN_I2V_MODELS + AUDIO_MODELS
MODELS_BY_KEY: Dict[str, I2VModelFile] = {m.key: m for m in ALL_MODELS}


# ----------------------------------------------------------------------------
# Verificación remota vía /object_info
# ----------------------------------------------------------------------------
# Mapea cada cargador de ComfyUI al campo de input que lista sus ficheros.
_LOADER_FILE_FIELDS = {
    "UnetLoaderGGUF": "unet_name",
    "UnetLoaderGGUFDisTorch2MultiGPU": "unet_name",
    "UnetLoaderGGUFAdvancedDisTorch2MultiGPU": "unet_name",
    "UNETLoader": "unet_name",
    "VAELoader": "vae_name",
    "CLIPLoader": "clip_name",
    "CLIPLoaderGGUF": "clip_name",
    "CheckpointLoaderSimple": "ckpt_name",
}


def _available_files(object_info: dict) -> Dict[str, set]:
    """Para cada campo de cargador, el conjunto de ficheros que ComfyUI ofrece."""
    out: Dict[str, set] = {}
    for node, field in _LOADER_FILE_FIELDS.items():
        spec = object_info.get(node)
        if not spec:
            continue
        try:
            choices = spec["input"]["required"][field][0]
        except Exception:
            continue
        if isinstance(choices, list):
            # Los nombres pueden venir con subcarpeta (p.ej. "HighNoise/foo.gguf").
            names = set()
            for c in choices:
                names.add(c)
                names.add(Path(str(c)).name)
            out.setdefault(field, set()).update(names)
    return out


def model_report(object_info: Optional[dict], include_audio: bool = True) -> List[dict]:
    """Devuelve, por modelo, si ComfyUI lo ve (a partir de ``/object_info``).

    Si ``object_info`` es None (ComfyUI no responde) marca todo como desconocido.
    """
    files = _available_files(object_info or {})
    # Campo de cargador que esperaríamos para cada subcarpeta.
    field_for_subfolder = {
        "unet": "unet_name",
        "vae": "vae_name",
        "text_encoders": "clip_name",
        "checkpoints": "ckpt_name",
    }
    report = []
    catalog = ALL_MODELS if include_audio else WAN_I2V_MODELS
    for m in catalog:
        field = field_for_subfolder.get(m.subfolder)
        present: Optional[bool]
        if object_info is None or not files:
            present = None
        elif field is None:
            present = None
        else:
            avail = files.get(field, set())
            present = m.filename in avail or Path(m.filename).name in avail
        report.append({"model": m, "present": present})
    return report


# ----------------------------------------------------------------------------
# Descarga local (para scripts/setup_i2v.py)
# ----------------------------------------------------------------------------
def comfy_models_root(comfy_path: str) -> Optional[Path]:
    """``<comfy_path>/models`` si existe; None si la ruta no parece ComfyUI."""
    if not comfy_path:
        return None
    root = Path(comfy_path).expanduser().resolve()
    models = root / "models"
    if (root / "main.py").exists() or models.exists():
        return models
    return None


def _download(url: str, dest: Path, token: Optional[str], progress: ProgressCb) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "fuser-i2v/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as out:
            while True:
                buf = resp.read(1 << 20)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if progress and total:
                    progress(done / total, f"Descargando {dest.name}")
    tmp.replace(dest)


def download_models(
    models_root: Path,
    *,
    include_audio: bool = True,
    token: Optional[str] = None,
    progress: ProgressCb = None,
) -> List[str]:
    """Descarga los modelos que falten a ``<models_root>/<subfolder>/``.

    Devuelve una lista de mensajes (errores / saltos por gating). Los ficheros
    *gated* (Stable Audio) se saltan si no hay token, con instrucciones.
    """
    msgs: List[str] = []
    catalog = ALL_MODELS if include_audio else WAN_I2V_MODELS
    for m in catalog:
        dest = models_root / m.subfolder / m.filename
        if dest.exists() and dest.stat().st_size > 0:
            log.info("Ya está: %s", dest)
            continue
        if m.gated and not token:
            msgs.append(
                f"⏭️  {m.filename}: requiere aceptar la licencia en "
                f"https://huggingface.co/{m.repo} y exportar HF_TOKEN. Sáltalo o "
                f"descárgalo a mano en {dest.parent}."
            )
            continue
        try:
            log.info("Descargando %s -> %s", m.url, dest)
            _download(m.url, dest, token, progress)
        except Exception as exc:  # pragma: no cover - depende de la red
            msgs.append(f"❌ {m.filename}: {exc}  (URL: {m.url})")
    return msgs
