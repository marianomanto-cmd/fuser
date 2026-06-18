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
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logger
