"""Gestión de memoria VRAM/RAM y configuración de execution providers.

Este es el cerebro de la optimización para 8 GB de VRAM. Decide:

1. Qué *execution provider* usa cada modelo (CUDA vs CPU).
2. El **límite de arena de VRAM** por sesión de onnxruntime (``gpu_mem_limit``),
   evitando que un modelo acapare toda la VRAM.
3. La estrategia de crecimiento de arena (``kSameAsRequested``) y la búsqueda
   de algoritmos de convolución (``HEURISTIC``) para minimizar picos de VRAM.
4. El **offloading** del enhancer a CPU/RAM en los modos de baja VRAM.
5. El tamaño de los buffers en RAM (prefetch de frames y cola de escritura),
   aprovechando los 40 GB de RAM para solapar I/O con cómputo en GPU.

Referencia de tuning: las mismas claves que usan FaceFusion/Roop para correr
inswapper en GPUs modestas, expuestas aquí de forma explícita y configurable.
"""
from __future__ import annotations

from typing import List

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.system import get_system_info

log = get_logger(__name__)

GIB = 1024**3


class MemoryManager:
    """Construye providers/opciones a partir de los ``Settings`` resueltos."""

    def __init__(self, settings: Settings):
        self.settings = settings.resolved()
        self.info = get_system_info()
        self.use_gpu = self.info.has_cuda and not self.settings.force_cpu
        if not self.use_gpu and not self.settings.force_cpu:
            log.warning("Sin CUDA: todo correrá en CPU (lento; usar solo para probar la UI).")

    # ----- Execution providers -------------------------------------------------
    def _cuda_options(self) -> dict:
        # onnxruntime exige que los valores de las opciones de provider sean
        # strings, así que serializamos todo a str para máxima compatibilidad.
        return {
            "device_id": "0",
            # Solo crece la arena cuando se pide -> menos VRAM reservada de más.
            "arena_extend_strategy": "kSameAsRequested",
            # Techo de VRAM por sesión: evita que un modelo se coma toda la tarjeta.
            "gpu_mem_limit": str(int(self.settings.gpu_mem_limit_gb * GIB)),
            # HEURISTIC evita el "warm-up" exhaustivo que dispara la VRAM.
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": "1",
        }

    def _gpu_providers(self) -> List:
        # CPU como fallback automático si una op no cabe/!soporta en GPU.
        return [("CUDAExecutionProvider", self._cuda_options()), "CPUExecutionProvider"]

    def _cpu_providers(self) -> List:
        return ["CPUExecutionProvider"]

    def analyser_providers(self) -> List:
        return self._gpu_providers() if self.use_gpu else self._cpu_providers()

    def swapper_providers(self) -> List:
        return self._gpu_providers() if self.use_gpu else self._cpu_providers()

    def enhancer_providers(self) -> List:
        """El enhancer se mueve a CPU/RAM en los modos de baja VRAM (offloading)."""
        if not self.use_gpu:
            return self._cpu_providers()
        if self.settings.preset.get("enhancer_device") == "cpu":
            log.info("Offloading del enhancer a CPU/RAM para liberar VRAM.")
            return self._cpu_providers()
        return self._gpu_providers()

    def ctx_id(self) -> int:
        """ctx_id para InsightFace: 0 = GPU, -1 = CPU."""
        return 0 if self.use_gpu else -1

    # ----- Opciones de sesión (hilos / RAM) -----------------------------------
    def session_options(self, on_cpu: bool):
        """SessionOptions de onnxruntime; en CPU usa todos los hilos disponibles."""
        try:
            import onnxruntime as ort
        except Exception:
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = True
        opts.enable_mem_pattern = True
        if on_cpu:
            # Aprovecha la CPU/RAM del sistema cuando el modelo corre en CPU.
            opts.intra_op_num_threads = max(1, self.info.cpu_count)
        return opts

    # ----- Dimensionado de buffers en RAM -------------------------------------
    @property
    def prefetch_frames(self) -> int:
        return int(self.settings.preset["prefetch_frames"])

    @property
    def writer_queue(self) -> int:
        return int(self.settings.preset["writer_queue"])

    @property
    def det_size(self) -> int:
        return int(self.settings.det_size)

    def buffer_sizes(self, frame_shape) -> tuple:
        """Tamaño de las colas de RAM (prefetch, escritura).

        Con ``ram_boost`` se dimensionan según la **RAM libre real**: se reserva
        ~30% de la RAM disponible para los buffers de frames (repartida entre las
        dos colas). Así, en una máquina con 40 GB, la GPU casi nunca espera por el
        disco. Nunca baja del valor del preset y tiene un tope de seguridad.
        """
        base_pf, base_wq = self.prefetch_frames, self.writer_queue
        if not self.settings.ram_boost or not self.info.ram_available_gb:
            return base_pf, base_wq
        h, w = frame_shape[:2]
        frame_mb = (h * w * 3) / (1024 ** 2)
        if frame_mb <= 0:
            return base_pf, base_wq
        budget_mb = self.info.ram_available_gb * 1024 * 0.30
        n = int(budget_mb / frame_mb / 2)
        n = max(base_pf, min(n, 800))
        return n, n

    def two_pass_chunk(self, frame_shape) -> int:
        """Nº de frames que cabe procesar por tramo en modo 2 pasadas (acota RAM).

        El modo de 2 pasadas guarda un tramo de frames en RAM para suavizar los
        landmarks con ventana centrada. El tamaño del tramo se ajusta a la RAM
        libre (~45%), con límites prudentes.
        """
        if not self.info.ram_available_gb:
            return 300
        h, w = frame_shape[:2]
        frame_mb = max((h * w * 3) / (1024 ** 2), 0.1)
        budget_mb = self.info.ram_available_gb * 1024 * 0.45
        n = int(budget_mb / frame_mb)
        return max(60, min(n, 4000))

    def summary(self) -> str:
        dev = "GPU (CUDA)" if self.use_gpu else "CPU"
        enh = "CPU/RAM" if self.enhancer_providers() == self._cpu_providers() else dev
        boost = "ON" if self.settings.ram_boost else "OFF"
        return (
            f"Cómputo: {dev} | Enhancer: {enh} | "
            f"Límite VRAM/sesión: {self.settings.gpu_mem_limit_gb:.1f} GB | "
            f"det_size: {self.det_size} | RAM-boost: {boost}"
        )
