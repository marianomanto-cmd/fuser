"""Punto de entrada de Fuser.

Lanza la interfaz Gradio. Compatible con ejecución local y con Hugging Face
Spaces (que importa este archivo y usa la variable ``demo`` o ejecuta ``app.py``).

Uso:
    python app.py                      # local, http://127.0.0.1:7860
    python app.py --share              # crea un enlace público temporal
    python app.py --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import argparse
import os

from fuser import __app_name__, __version__
from fuser.config import ensure_dirs
from fuser.ui import build_interface
from fuser.utils.logging import get_logger

log = get_logger("fuser")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{__app_name__} · Face swap de vídeo local")
    p.add_argument("--host", default=os.environ.get("FUSER_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("FUSER_PORT", "7860")))
    p.add_argument("--share", action="store_true", help="Crear enlace público de Gradio")
    p.add_argument("--listen", action="store_true", help="Escuchar en 0.0.0.0 (LAN)")
    return p.parse_args()


# ``demo`` a nivel de módulo: requerido por Hugging Face Spaces.
ensure_dirs()
demo = build_interface()
demo.queue(max_size=8)  # cola para que el progreso y la concurrencia funcionen bien


def main() -> None:
    args = parse_args()
    # En Hugging Face Spaces hay que escuchar en 0.0.0.0 (variable SPACE_ID presente).
    on_spaces = bool(os.environ.get("SPACE_ID"))
    host = "0.0.0.0" if (args.listen or on_spaces) else args.host
    log.info("Iniciando %s v%s en http://%s:%d", __app_name__, __version__, host, args.port)
    # Icono de la app (pestaña/ventana). El .ico vive en la raíz del proyecto.
    icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fuser.ico")
    demo.launch(server_name=host, server_port=args.port, share=args.share, show_error=True,
                favicon_path=icon if os.path.isfile(icon) else None)


if __name__ == "__main__":
    main()
