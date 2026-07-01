"""Módulo *Imagen → Vídeo* de Fuser (Wan 2.2 14B I2V vía ComfyUI).

Esta es una **función nueva e independiente** del face swap: NO toca el pipeline
de caras. Genera un vídeo corto (~480p, ~6 s, con audio) a partir de **una imagen
+ un prompt de texto**, usando un servidor **ComfyUI local** como motor de
inferencia (Wan 2.2 I2V con cuantización GGUF y *offloading* a RAM, pensado para
**8 GB de VRAM + 40 GB de RAM**).

Arquitectura (resumen):

    UI (pestaña Gradio)  ->  I2VService  ->  ComfyUIClient (HTTP + WebSocket)
                                   |             |
                                   |             +-- ComfyUI (proceso aparte, :8188)
                                   |                 · Wan 2.2 I2V GGUF (vídeo sin audio)
                                   |                 · Stable Audio Open (audio)
                                   +-- ffmpeg (mezcla vídeo + audio)

Todo el código de este paquete usa **solo la librería estándar** para hablar con
ComfyUI (``urllib``); ``websocket-client`` es **opcional** (mejora la barra de
progreso, con *fallback* a *polling* si no está). Así la app principal sigue
arrancando sin dependencias nuevas obligatorias.
"""
from __future__ import annotations

__all__ = ["config", "comfy_client", "workflow", "service", "models"]
