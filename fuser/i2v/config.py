"""Configuración del módulo *Imagen → Vídeo* (Wan 2.2 14B I2V vía ComfyUI).

Centraliza:
- Conexión con el servidor ComfyUI (URL, timeouts).
- ``I2VSettings``: parámetros de una generación (resolución, duración, pasos,
  modelos, *offloading*, audio).
- Presets pensados para **8 GB de VRAM + 40 GB de RAM**.
- Catálogo de plantillas de workflow (ficheros JSON en ``workflows/``).

Como el resto de ``fuser.config``, este módulo NO importa dependencias pesadas:
puede usarse para construir la UI sin coste.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from ..config import OUTPUTS_DIR, PROJECT_ROOT, TEMP_DIR  # noqa: F401 (reexport útil)

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
WORKFLOWS_DIR = PACKAGE_DIR / "workflows"
# Carpeta donde la app deja los vídeos generados (con audio ya mezclado).
I2V_OUTPUT_DIR = Path(os.environ.get("FUSER_I2V_OUTPUT_DIR", OUTPUTS_DIR / "i2v")).resolve()


def ensure_i2v_dirs() -> None:
    I2V_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Conexión con ComfyUI
# ----------------------------------------------------------------------------
DEFAULT_COMFY_URL = os.environ.get("FUSER_COMFY_URL", "http://127.0.0.1:8188")
# Ruta opcional a una instalación de ComfyUI (para el lanzador automático y para
# que el script doctor sepa dónde están sus carpetas de modelos).
COMFY_PATH = os.environ.get("FUSER_COMFY_PATH", "")

# Tiempo máximo (s) que esperamos a que ComfyUI termine una generación de vídeo.
# El I2V de 14B en 8 GB es LENTO: damos margen amplio (configurable por entorno).
GENERATION_TIMEOUT_S = int(os.environ.get("FUSER_I2V_TIMEOUT", "3600"))


# ----------------------------------------------------------------------------
# Plantillas de workflow (API format de ComfyUI)
# ----------------------------------------------------------------------------
# Cada plantilla es un grafo en "API format" que la app carga y *parchea* por
# ``class_type`` (no por id), de modo que también funciona si el usuario exporta
# su propio workflow desde ComfyUI. Ver ``workflow.py``.
WF_WAN22_I2V_GGUF = "wan22_i2v_14b_gguf"            # GGUF puro + --lowvram (recomendado)
WF_WAN22_I2V_DISTORCH = "wan22_i2v_14b_gguf_distorch"  # GGUF + DisTorch2 (offload a RAM máximo)
WF_STABLE_AUDIO = "stable_audio"                     # texto -> audio (~6 s)

WORKFLOW_LABELS: Dict[str, str] = {
    WF_WAN22_I2V_GGUF: "Wan 2.2 14B I2V · GGUF (recomendado, --lowvram)",
    WF_WAN22_I2V_DISTORCH: "Wan 2.2 14B I2V · GGUF + DisTorch2 (offload máximo a RAM)",
}


# ----------------------------------------------------------------------------
# Presets de offloading para 8 GB de VRAM
# ----------------------------------------------------------------------------
# El *offloading* real de Wan 14B a la RAM ocurre en DOS sitios:
#   1) Flags de arranque de ComfyUI (--lowvram / --novram / --reserve-vram).
#   2) El nodo cargador GGUF: el cargador normal deja que la "smart memory" de
#      ComfyUI mueva pesos a RAM; el cargador DisTorch2 te deja fijar cuántos GB
#      "virtuales" tomar de la RAM (``virtual_vram_gb``).
# Estos presets recomiendan ambas cosas; la app aplica (2) parcheando el grafo y
# (1) la muestra en la UI / el lanzador.
OFFLOAD_BALANCED = "balanced_8gb"
OFFLOAD_MAX = "max_offload"
OFFLOAD_PERFORMANCE = "performance"

OFFLOAD_PRESETS: Dict[str, dict] = {
    # Equilibrado: lo más estable para 8 GB. GGUF Q4_K_M + --lowvram. La smart
    # memory de ComfyUI descarga sola lo que no cabe.
    OFFLOAD_BALANCED: dict(
        workflow=WF_WAN22_I2V_GGUF,
        comfy_flags=["--lowvram", "--reserve-vram", "0.6"],
        virtual_vram_gb=0.0,   # no aplica (cargador GGUF normal)
        note="Recomendado. GGUF Q4_K_M + --lowvram; ComfyUI descarga a RAM lo que no cabe.",
    ),
    # Offload máximo: usa ComfyUI-MultiGPU (DisTorch2) para empujar capas a la
    # RAM de forma explícita. Más lento pero el que menos peta en 8 GB.
    OFFLOAD_MAX: dict(
        workflow=WF_WAN22_I2V_DISTORCH,
        comfy_flags=["--lowvram", "--reserve-vram", "0.8"],
        virtual_vram_gb=6.0,   # cuántos GB "tomar prestados" de la RAM por modelo
        note="Máxima estabilidad en 8 GB: DisTorch2 mueve ~6 GB de pesos a la RAM por experto.",
    ),
    # Rendimiento: para quien tenga algo más de margen (deja más en VRAM).
    OFFLOAD_PERFORMANCE: dict(
        workflow=WF_WAN22_I2V_GGUF,
        comfy_flags=["--reserve-vram", "0.5"],
        virtual_vram_gb=0.0,
        note="Más rápido pero más arriesgado en 8 GB; ideal si tienes 10-12 GB.",
    ),
}

OFFLOAD_LABELS: Dict[str, str] = {
    "⚖️ Equilibrado (recomendado, 8 GB)": OFFLOAD_BALANCED,
    "🧠 Offload máximo a RAM (DisTorch2, 8 GB justo)": OFFLOAD_MAX,
    "⚡ Rendimiento (10-12 GB)": OFFLOAD_PERFORMANCE,
}


# ----------------------------------------------------------------------------
# Opciones de generación (para la UI)
# ----------------------------------------------------------------------------
# Wan trabaja mejor con lados múltiplos de 16. 480p ≈ 832x480 (16:9) o 480x832.
RESOLUTION_CHOICES = [
    ("480p horizontal 16:9 (832×480)", "832x480"),
    ("480p vertical 9:16 (480×832)", "480x832"),
    ("480p cuadrado (640×640)", "640x640"),
    ("640×480 4:3", "640x480"),
    ("512p horizontal (832×512)", "832x512"),
]

# Wan 2.2 trabaja a 16 fps. La longitud debe ser 4n+1 frames.
#   97 frames / 16 fps = 6.06 s  (≈ los 6 s pedidos)
DURATION_CHOICES = [
    ("≈4 s (65 frames)", 65),
    ("≈5 s (81 frames)", 81),
    ("≈6 s (97 frames) — recomendado", 97),
    ("≈7 s (113 frames)", 113),
]

SAMPLER_CHOICES = ["euler", "euler_ancestral", "dpmpp_2m", "uni_pc", "lcm"]
SCHEDULER_CHOICES = ["simple", "normal", "beta", "ddim_uniform", "karras"]

QUANT_NOTE = (
    "GGUF Q4_K_M (~9.6 GB por experto) es el mejor equilibrio para 8 GB. "
    "Si te sigue faltando memoria, baja a Q3_K_M; si te sobra RAM y quieres más "
    "calidad, sube a Q5_K_M (más lento)."
)

# Prompt negativo por defecto de Wan (oficial; mezcla ZH/EN). Evita artefactos
# típicos: sobreexposición, manos mal dibujadas, fondo recargado, etc.
WAN_DEFAULT_NEGATIVE = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, "
    "最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, "
    "画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, "
    "杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

DEFAULT_AUDIO_NEGATIVE = "noise, hiss, distortion, low quality"


# ----------------------------------------------------------------------------
# Settings de una generación
# ----------------------------------------------------------------------------
@dataclass
class I2VSettings:
    """Parámetros de un trabajo *Imagen → Vídeo*."""

    # --- Conexión ---
    comfy_url: str = DEFAULT_COMFY_URL

    # --- Offload / workflow ---
    offload_preset: str = OFFLOAD_BALANCED
    workflow: str = WF_WAN22_I2V_GGUF      # se deriva del preset si se deja por defecto
    virtual_vram_gb: float = 0.0           # se deriva del preset (solo DisTorch2)

    # --- Modelos (tal y como se llaman DENTRO de ComfyUI) ---
    high_noise_model: str = "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"
    low_noise_model: str = "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    text_encoder: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae: str = "wan_2.1_vae.safetensors"

    # --- Generación de vídeo ---
    width: int = 832
    height: int = 480
    length_frames: int = 97                # 4n+1 ; 97 ≈ 6 s a 16 fps
    fps: int = 16
    steps: int = 20                        # total (se reparte high/low a la mitad)
    cfg: float = 3.5
    sampler: str = "euler"
    scheduler: str = "simple"
    shift: float = 8.0                     # ModelSamplingSD3 (8.0 va bien a 480p)
    seed: int = -1                         # -1 = aleatoria

    # --- Audio (segundo paso, modelo aparte) ---
    audio_enabled: bool = True
    audio_prompt: str = ""                 # si vacío, se deriva del prompt de vídeo
    audio_negative: str = DEFAULT_AUDIO_NEGATIVE
    audio_seconds: float = 0.0             # 0 = igualar a la duración del vídeo
    audio_steps: int = 50
    audio_cfg: float = 4.5
    audio_seed: int = -1

    def resolved(self) -> "I2VSettings":
        """Rellena ``workflow`` y ``virtual_vram_gb`` desde el preset de offload."""
        out = I2VSettings(**asdict(self))
        preset = OFFLOAD_PRESETS.get(self.offload_preset, OFFLOAD_PRESETS[OFFLOAD_BALANCED])
        # Solo derivamos si el usuario no los fijó a mano.
        if out.workflow == WF_WAN22_I2V_GGUF and self.offload_preset != OFFLOAD_BALANCED:
            out.workflow = preset["workflow"]
        if out.virtual_vram_gb <= 0:
            out.virtual_vram_gb = float(preset["virtual_vram_gb"])
        if out.audio_seconds <= 0:
            out.audio_seconds = round(out.length_frames / max(1, out.fps), 1)
        return out

    @property
    def comfy_flags(self) -> List[str]:
        preset = OFFLOAD_PRESETS.get(self.offload_preset, OFFLOAD_PRESETS[OFFLOAD_BALANCED])
        flags = list(preset["comfy_flags"])
        if self.virtual_vram_gb > 0 and "--lowvram" not in flags and "--novram" not in flags:
            flags.insert(0, "--lowvram")
        return flags

    def to_dict(self) -> dict:
        return asdict(self)
