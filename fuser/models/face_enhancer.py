"""Realce / restauración de cara como ONNX puro (GFPGAN, CodeFormer, GPEN...).

Ejecutar los enhancers como ONNX (en vez de los paquetes de PyTorch/basicsr)
mantiene la app ligera, evita conflictos de versiones y permite mover el
enhancer entre GPU y CPU con un simple cambio de providers (clave para el modo
"VRAM mínima": el enhancer corre en CPU usando RAM mientras el swap usa la GPU).

Todos estos modelos comparten convención de E/S:
- Entrada: cara alineada 512x512, RGB, normalizada a [-1, 1], NCHW.
- Salida: misma forma en [-1, 1].
- CodeFormer añade un segundo input escalar de "fidelidad".
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import ModelInfo
from ..utils.logging import get_logger

log = get_logger(__name__)


class FaceEnhancer:
    """Restaurador de caras genérico sobre onnxruntime."""

    def __init__(self, model_path: str, providers: list, info: ModelInfo, sess_options=None):
        self._path = str(model_path)
        self._providers = providers
        self._info = info
        self._sess_options = sess_options
        self._session = None
        self._inputs = None
        self._size = info.size

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort  # import perezoso

        log.info("Cargando enhancer %s en %s", self._info.key, self._providers)
        self._session = ort.InferenceSession(
            self._path, sess_options=self._sess_options, providers=self._providers
        )
        self._inputs = self._session.get_inputs()
        # Algunos modelos exponen su tamaño de entrada; si no, usamos el del registro.
        shape = self._inputs[0].shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4 and isinstance(shape[2], int):
            self._size = int(shape[2])

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _build_weight(self, fidelity: float) -> np.ndarray:
        winput = self._inputs[1]
        dtype = np.float64 if "double" in winput.type else np.float32
        dims = [d if isinstance(d, int) and d > 0 else 1 for d in (winput.shape or [])]
        if not dims:
            return np.array(fidelity, dtype=dtype)
        return np.full(dims, fidelity, dtype=dtype)

    def run(self, face_bgr: np.ndarray, fidelity: float = 0.7) -> np.ndarray:
        """Restaura una cara BGR. Devuelve la cara realzada al MISMO tamaño de entrada."""
        if self._session is None:
            self.load()
        src_h, src_w = face_bgr.shape[:2]

        resized = cv2.resize(face_bgr, (self._size, self._size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) / 0.5  # -> [-1, 1]
        blob = rgb.transpose(2, 0, 1)[None, ...]  # NCHW

        feeds = {self._inputs[0].name: blob}
        if self._info.has_weight_input and len(self._inputs) > 1:
            feeds[self._inputs[1].name] = self._build_weight(fidelity)

        output = self._session.run(None, feeds)[0]
        out = output[0].transpose(1, 2, 0)  # HWC, [-1, 1]
        out = (np.clip(out, -1, 1) + 1) / 2.0 * 255.0
        out = out.clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        if (src_h, src_w) != (self._size, self._size):
            out_bgr = cv2.resize(out_bgr, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
        return out_bgr

    def unload(self) -> None:
        self._session = None
