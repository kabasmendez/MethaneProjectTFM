#!/usr/bin/env python3
"""
Step00ProjectAudit.py

Auditoría inicial del proyecto.

Este script:
- valida RunTag ExpDDHHMM;
- crea la estructura estándar del experimento;
- valida ProjectConfig.yaml;
- valida ConfigA, ConfigB y ConfigC;
- valida VisualizationConfig.yaml;
- registra salidas en Audit/OutputIndex.csv;
- guarda ProjectAudit.json y ProjectAudit.log.

No accede al dataset pesado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml, ValidateFeatureConfig, ValidateProjectConfig
from Source.LoggingUtils import CreateLogger
from Source.Paths import CONFIG_NAMES, CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ResolveProjectPath, ValidateCommonArguments
from Source.VisualizationStyle import LoadVisualizationConfig, ResolveFontFamily


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Auditoría inicial de MethaneProjectTFM.")
    Parser = AddCommonArguments(Parser)
    Parser.add_argument(
        "--VisualizationConfig",
        default="Configs/VisualizationConfig.yaml",
        help="Ruta al archivo VisualizationConfig.yaml.",
    )
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    LogPath = Paths.LogsDirectory / "Step00ProjectAudit.log"
    Logger = CreateLogger("Step00ProjectAudit", LogPath)

    Logger.info("Iniciando auditoría del proyecto.")
    Logger.info("ProjectRoot: %s", Paths.ProjectRoot)
    Logger.info("RunTag: %s", Args.RunTag)
    Logger.info("RunDirectory: %s", Paths.RunDirectory)

    ProjectConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.ProjectConfig)
    ProjectConfig = LoadYaml(ProjectConfigPath)
    ValidateProjectConfig(ProjectConfig)

    VisualizationConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.VisualizationConfig)
    VisualizationConfig = LoadVisualizationConfig(VisualizationConfigPath)
    ResolvedFont = ResolveFontFamily(VisualizationConfig)

    FeatureConfigStatus = {}

    for ConfigName in CONFIG_NAMES:
        ConfigPath = Paths.ProjectRoot / "Configs" / f"{ConfigName}.yaml"
        Config = LoadYaml(ConfigPath)
        ValidateFeatureConfig(Config, ExpectedName=ConfigName)

        FeatureConfigStatus[ConfigName] = {
            "Path": str(ConfigPath),
            "InputChannels": int(Config["InputChannels"]),
            "FeatureCount": len(Config["Features"]),
            "RequiresContextMetadata": bool(Config["RequiresContextMetadata"]),
        }

    RequiredDirectories = [
        "Configs",
        "Scripts",
        "Source",
        "Tests",
        "App",
        "Notebooks",
        "Outputs",
        "Outputs/Experiments",
    ]

    DirectoryStatus = {}

    for RelativeDirectory in RequiredDirectories:
        Directory = Paths.ProjectRoot / RelativeDirectory
        DirectoryStatus[RelativeDirectory] = Directory.exists()

        if not Directory.exists():
            raise FileNotFoundError(f"Falta carpeta requerida: {Directory}")

    AuditPath = Paths.AuditDirectory / "ProjectAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"
    ManifestPath = Paths.AuditDirectory / "ExperimentManifest.json"

    Audit = BuildAuditRecord(
        ScriptName="Step00ProjectAudit.py",
        RunTag=Args.RunTag,
        Parameters={
            "ProjectConfig": str(ProjectConfigPath),
            "VisualizationConfig": str(VisualizationConfigPath),
        },
        Inputs={
            "ProjectConfig": str(ProjectConfigPath),
            "VisualizationConfig": str(VisualizationConfigPath),
            "FeatureConfigs": FeatureConfigStatus,
        },
        Outputs={
            "Audit": str(AuditPath),
            "Log": str(LogPath),
            "OutputIndex": str(OutputIndexPath),
            "ExperimentManifest": str(ManifestPath),
        },
        Status="Success",
        Details={
            "DirectoryStatus": DirectoryStatus,
            "FeatureConfigStatus": FeatureConfigStatus,
            "RequestedFont": VisualizationConfig["Visualization"].get("FontFamily"),
            "ResolvedFont": ResolvedFont,
            "RunDirectory": str(Paths.RunDirectory),
        },
    )

    Manifest = {
        "RunTag": Args.RunTag,
        "ProjectName": ProjectConfig["ProjectName"],
        "DatasetPath": str(
            Path(ProjectConfig["Dataset"]["DataRoot"]) / ProjectConfig["Dataset"]["DatasetName"]
        ),
        "Seed": ProjectConfig["Seed"],
        "Configs": list(CONFIG_NAMES),
        "OutputsRoot": str(Paths.RunDirectory),
        "Status": "Initialized",
    }

    WriteJson(Audit, AuditPath)
    WriteJson(Manifest, ManifestPath)

    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=Args.RunTag,
        Step="Step00ProjectAudit",
        Config="Project",
        Model="None",
        OutputType="Audit",
        RelativePath="Audit/ProjectAudit.json",
        Created=AuditPath.exists(),
        Description="Auditoría inicial del proyecto.",
    )

    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=Args.RunTag,
        Step="Step00ProjectAudit",
        Config="Project",
        Model="None",
        OutputType="Manifest",
        RelativePath="Audit/ExperimentManifest.json",
        Created=ManifestPath.exists(),
        Description="Manifiesto inicial del experimento.",
    )

    Logger.info("Auditoría completada correctamente.")
    Logger.info("Audit: %s", AuditPath)
    Logger.info("Manifest: %s", ManifestPath)
    Logger.info("OutputIndex: %s", OutputIndexPath)


if __name__ == "__main__":
    Main()
