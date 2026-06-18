"""Fuser — Face swap de video local, optimizado para 8 GB de VRAM + 40 GB de RAM.

El paquete está organizado en cuatro capas:

- ``fuser.config``  : configuración, presets de memoria y registro de modelos.
- ``fuser.utils``   : utilidades de sistema, vídeo e imagen (sin dependencias pesadas).
- ``fuser.models``  : envoltorios ONNX de detección, swap y realce de caras.
- ``fuser.core``    : gestión de memoria y orquestación del pipeline.
- ``fuser.ui``      : interfaz Gradio.

Las dependencias pesadas (onnxruntime / insightface) se importan de forma
perezosa dentro de ``fuser.models`` para que la UI pueda arrancar al instante
incluso sin los modelos descargados.
"""

__version__ = "1.7.0"
__app_name__ = "Fuser"

__all__ = ["__version__", "__app_name__"]
