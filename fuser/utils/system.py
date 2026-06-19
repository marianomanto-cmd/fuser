"""Detección de hardware y proveedores de ejecución.

Se usa para:
- Saber si hay CUDA disponible (y elegir GPU vs CPU).
- Estimar VRAM y RAM para dimensionar buffers y límites de memoria.
- Mostrar un resumen del sistema en la UI.

Todas las importaciones potencialmente ausentes (onnxruntime, psutil, pynvml)
son perezosas y tolerantes a fallos: si algo no está, se degrada con elegancia.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from .logging import get_logger

log = get_logger(__name__)


@dataclass
class SystemInfo:
    has_cuda: bool                # True si hay aceleración GPU (CUDA o DirectML)
    gpu_provider: Optional[str]   # EP de GPU: CUDAExecutionProvider / DmlExecutionProvider / None
    providers: List[str]
    gpu_name: Optional[str]
    vram_total_gb: Optional[float]
    vram_free_gb: Optional[float]
    ram_total_gb: Optional[float]
    ram_available_gb: Optional[float]
    cpu_count: int
    ffmpeg_available: bool


@lru_cache(maxsize=1)
def available_providers() -> List[str]:
    """Lista de execution providers de onnxruntime disponibles."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover - depende del entorno
        log.warning("onnxruntime no disponible: %s", exc)
        return []


def has_cuda() -> bool:
    return "CUDAExecutionProvider" in available_providers()


def _query_nvml() -> Optional[dict]:
    """Consulta nombre y memoria de la GPU vía pynvml si está instalado."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "ignore")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "name": name,
            "total_gb": mem.total / (1024**3),
            "free_gb": mem.free / (1024**3),
        }
    except Exception:
        return None


def _query_ram() -> tuple[Optional[float], Optional[float]]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / (1024**3), vm.available / (1024**3)
    except Exception:
        return None, None


def ffmpeg_path() -> Optional[str]:
    """Localiza un binario de ffmpeg: del sistema o el que trae imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_system_info() -> SystemInfo:
    """Recopila un retrato del sistema (cacheado)."""
    providers = available_providers()
    has_cuda_ep = "CUDAExecutionProvider" in providers
    has_dml_ep = "DmlExecutionProvider" in providers
    gpu_provider = (
        "CUDAExecutionProvider" if has_cuda_ep
        else "DmlExecutionProvider" if has_dml_ep
        else None
    )
    nvml = _query_nvml() if gpu_provider else None
    ram_total, ram_avail = _query_ram()
    return SystemInfo(
        has_cuda=gpu_provider is not None,
        gpu_provider=gpu_provider,
        providers=providers,
        gpu_name=(nvml or {}).get("name"),
        vram_total_gb=(nvml or {}).get("total_gb"),
        vram_free_gb=(nvml or {}).get("free_gb"),
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        cpu_count=os.cpu_count() or 4,
        ffmpeg_available=ffmpeg_path() is not None,
    )


def format_system_summary() -> str:
    """Resumen en Markdown para mostrar en la UI."""
    info = get_system_info()
    lines = ["### 🖥️ Estado del sistema", ""]
    if info.has_cuda:
        backend = "DirectML" if info.gpu_provider == "DmlExecutionProvider" else "CUDA"
        gpu = info.gpu_name or f"GPU ({backend})"
        vram = (
            f"{info.vram_free_gb:.1f} GB libres / {info.vram_total_gb:.1f} GB"
            if info.vram_total_gb
            else "VRAM no consultable (instala `pynvml`)"
        )
        lines.append(f"- **GPU:** {gpu}")
        lines.append(f"- **VRAM:** {vram}")
        lines.append(f"- **Aceleración:** ✅ {backend}")
    else:
        lines.append("- **GPU:** ❌ Sin aceleración GPU — se usará **CPU** (lento, solo para probar la UI)")

    if info.ram_total_gb:
        lines.append(
            f"- **RAM:** {info.ram_available_gb:.1f} GB libres / {info.ram_total_gb:.1f} GB"
        )
    lines.append(f"- **CPU:** {info.cpu_count} hilos")
    lines.append(f"- **FFmpeg:** {'✅' if info.ffmpeg_available else '❌ no encontrado'}")
    lines.append(f"- **Providers ONNX:** `{', '.join(info.providers) or 'ninguno'}`")
    return "\n".join(lines)
