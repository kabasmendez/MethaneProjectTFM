"""
ConfigUtils.py

Lectura y validación de archivos YAML del proyecto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def LoadYaml(ConfigPath: str | Path) -> dict[str, Any]:
    """Carga un YAML como diccionario."""
    ConfigPath = Path(ConfigPath)

    if not ConfigPath.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {ConfigPath}")

    with ConfigPath.open("r", encoding="utf-8") as File:
        Config = yaml.safe_load(File)

    if not isinstance(Config, dict):
        raise ValueError(f"El YAML no contiene un diccionario válido: {ConfigPath}")

    return Config


def RequireKeys(Data: dict[str, Any], RequiredKeys: list[str], Context: str) -> None:
    """Valida que existan claves obligatorias."""
    MissingKeys = [Key for Key in RequiredKeys if Key not in Data]

    if MissingKeys:
        raise KeyError(f"Faltan claves obligatorias en {Context}: {MissingKeys}")


def ValidateProjectConfig(ProjectConfig: dict[str, Any]) -> None:
    """Valida ProjectConfig.yaml."""
    RequireKeys(
        ProjectConfig,
        ["ProjectName", "Seed", "Dataset", "Outputs", "Filters", "Split", "ExpectedShapes"],
        "ProjectConfig",
    )

    RequireKeys(ProjectConfig["Dataset"], ["DataRoot", "DatasetName"], "ProjectConfig.Dataset")
    RequireKeys(ProjectConfig["Outputs"], ["Root"], "ProjectConfig.Outputs")
    RequireKeys(ProjectConfig["Split"], ["Train", "Validation", "Test"], "ProjectConfig.Split")

    Split = ProjectConfig["Split"]
    Total = float(Split["Train"]) + float(Split["Validation"]) + float(Split["Test"])

    if abs(Total - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones de Split deben sumar 1.0. Suma actual: {Total}")


def ValidateFeatureConfig(FeatureConfig: dict[str, Any], ExpectedName: str | None = None) -> None:
    """Valida ConfigA.yaml, ConfigB.yaml o ConfigC.yaml."""
    RequireKeys(
        FeatureConfig,
        ["FeatureConfig", "InputChannels", "Features", "RequiresContextMetadata"],
        "FeatureConfig",
    )

    ConfigName = FeatureConfig["FeatureConfig"]

    if ExpectedName is not None and ConfigName != ExpectedName:
        raise ValueError(f"Se esperaba {ExpectedName}, pero el YAML declara {ConfigName}")

    Features = FeatureConfig["Features"]

    if not isinstance(Features, list) or len(Features) == 0:
        raise ValueError(f"{ConfigName}: Features debe ser una lista no vacía.")

    InputChannels = int(FeatureConfig["InputChannels"])

    if InputChannels != len(Features):
        raise ValueError(
            f"{ConfigName}: InputChannels={InputChannels}, "
            f"pero Features contiene {len(Features)} entradas."
        )
