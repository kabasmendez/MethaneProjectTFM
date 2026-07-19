"""
RunUtils.py

Utilidades comunes para argumentos CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Source.Paths import ValidateRunTag


def AddCommonArguments(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Añade argumentos comunes a todos los scripts."""
    Parser.add_argument("--RunTag", required=True, help="Identificador del experimento: ExpDDHHMM.")
    Parser.add_argument(
        "--ProjectConfig",
        default="Configs/ProjectConfig.yaml",
        help="Ruta al archivo ProjectConfig.yaml.",
    )
    return Parser


def AddFeatureConfigArgument(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Añade argumento de configuración de features."""
    Parser.add_argument(
        "--FeatureConfig",
        required=True,
        choices=["ConfigA", "ConfigB", "ConfigC"],
        help="Configuración de features.",
    )
    return Parser


def ResolveProjectPath(ProjectRoot: Path, PathValue: str | Path) -> Path:
    """Resuelve ruta absoluta o relativa a la raíz del proyecto."""
    PathObject = Path(PathValue)

    if PathObject.is_absolute():
        return PathObject

    return ProjectRoot / PathObject


def ValidateCommonArguments(Args: argparse.Namespace) -> None:
    """Valida argumentos comunes."""
    ValidateRunTag(Args.RunTag)
