"""
Paths.py

Gestión centralizada de rutas para MethaneProjectTFM.

Responsabilidades:
- Validar RunTag con formato ExpDDHHMM.
- Resolver la raíz del proyecto.
- Crear la estructura estándar de cada experimento.
- Crear carpetas ConfigA, ConfigB, ConfigC, Compare, Reports, Logs y Audit.

Regla importante:
Este módulo no genera timestamps. El RunTag debe recibirse desde línea de comandos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


VALID_RUN_TAG_PATTERN = re.compile(r"^Exp[0-9]{6}$")

CONFIG_NAMES = ("ConfigA", "ConfigB", "ConfigC")
CONFIG_SUBDIRECTORIES = ("Tables", "Metrics", "Figures", "Checkpoints", "Features", "Audit")
COMPARE_SUBDIRECTORIES = ("Tables", "Metrics", "Figures")


@dataclass(frozen=True)
class ExperimentPaths:
    ProjectRoot: Path
    RunTag: str
    RunDirectory: Path
    TablesDirectory: Path
    LogsDirectory: Path
    AuditDirectory: Path
    ReportsDirectory: Path
    FiguresDirectory: Path
    CompareDirectory: Path


def GetProjectRoot() -> Path:
    """Devuelve la raíz del proyecto."""
    return Path(__file__).resolve().parents[1]


def ValidateRunTag(RunTag: str) -> str:
    """Valida que RunTag cumpla el formato ExpDDHHMM."""
    if not isinstance(RunTag, str):
        raise TypeError("RunTag debe ser str.")

    if not VALID_RUN_TAG_PATTERN.match(RunTag):
        raise ValueError(
            f"RunTag inválido: {RunTag}. "
            "Formato esperado: ExpDDHHMM, por ejemplo Exp241930."
        )

    return RunTag


def GetRunDirectory(RunTag: str) -> Path:
    """Devuelve la carpeta del experimento sin crearla."""
    ValidateRunTag(RunTag)
    return GetProjectRoot() / "Outputs" / "Experiments" / RunTag


def CreateExperimentDirectories(RunTag: str) -> ExperimentPaths:
    """Crea la estructura estándar de carpetas de un experimento."""
    ValidateRunTag(RunTag)

    ProjectRoot = GetProjectRoot()
    RunDirectory = ProjectRoot / "Outputs" / "Experiments" / RunTag

    TablesDirectory = RunDirectory / "Tables"
    LogsDirectory = RunDirectory / "Logs"
    AuditDirectory = RunDirectory / "Audit"
    ReportsDirectory = RunDirectory / "Reports"
    FiguresDirectory = RunDirectory / "Figures"
    CompareDirectory = RunDirectory / "Compare"

    for Directory in [
        TablesDirectory,
        LogsDirectory,
        AuditDirectory,
        ReportsDirectory,
        FiguresDirectory,
        CompareDirectory,
    ]:
        Directory.mkdir(parents=True, exist_ok=True)

    for SubdirectoryName in COMPARE_SUBDIRECTORIES:
        (CompareDirectory / SubdirectoryName).mkdir(parents=True, exist_ok=True)

    for ConfigName in CONFIG_NAMES:
        ConfigDirectory = RunDirectory / ConfigName
        for SubdirectoryName in CONFIG_SUBDIRECTORIES:
            (ConfigDirectory / SubdirectoryName).mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        ProjectRoot=ProjectRoot,
        RunTag=RunTag,
        RunDirectory=RunDirectory,
        TablesDirectory=TablesDirectory,
        LogsDirectory=LogsDirectory,
        AuditDirectory=AuditDirectory,
        ReportsDirectory=ReportsDirectory,
        FiguresDirectory=FiguresDirectory,
        CompareDirectory=CompareDirectory,
    )


def GetConfigDirectory(RunTag: str, FeatureConfig: str) -> Path:
    """Devuelve la carpeta de una configuración dentro de un experimento."""
    ValidateRunTag(RunTag)

    if FeatureConfig not in CONFIG_NAMES:
        raise ValueError(f"FeatureConfig inválida: {FeatureConfig}. Opciones: {CONFIG_NAMES}")

    return GetRunDirectory(RunTag) / FeatureConfig
