"""Logging sencillo y consistente para toda la aplicación."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "fuser") -> logging.Logger:
    """Devuelve un logger configurado una sola vez con formato legible."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("fuser")
        root.addHandler(handler)
        # Además, a un archivo en la raíz del repo, para poder revisar la sesión
        # después de un crash (la consola es efímera). Se sobrescribe en cada arranque.
        try:
            from pathlib import Path

            logfile = Path(__file__).resolve().parents[2] / "fuser_session.log"
            fh = logging.FileHandler(str(logfile), mode="w", encoding="utf-8")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            root.addHandler(fh)
        except Exception:
            pass
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logger
