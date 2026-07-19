"""
LoggingUtils.py

Configuración estándar de logs para scripts del proyecto.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def CreateLogger(LoggerName: str, LogPath: Path) -> logging.Logger:
    """Crea logger con salida a consola y archivo."""
    LogPath.parent.mkdir(parents=True, exist_ok=True)

    Logger = logging.getLogger(LoggerName)
    Logger.setLevel(logging.INFO)
    Logger.handlers.clear()
    Logger.propagate = False

    Formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    StreamHandler = logging.StreamHandler(sys.stdout)
    StreamHandler.setFormatter(Formatter)
    StreamHandler.setLevel(logging.INFO)

    FileHandler = logging.FileHandler(LogPath, mode="w", encoding="utf-8")
    FileHandler.setFormatter(Formatter)
    FileHandler.setLevel(logging.INFO)

    Logger.addHandler(StreamHandler)
    Logger.addHandler(FileHandler)

    return Logger
