#!/usr/bin/env python3
"""
Step02InspectMetadata.py

Inspecciona metadatos reales del DatasetFiltered para decidir si ConfigC puede
construirse con viento y ángulo solar.

Salidas:
- Tables/MetadataColumns.csv
- Tables/ContextMetadataCheck.csv
- Audit/InspectMetadataAudit.json
- Logs/Step02InspectMetadata.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml, ValidateProjectConfig
from Source.ContextFeatures import BuildContextSummary, FindContextColumnCandidates
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ResolveProjectPath, ValidateCommonArguments


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Inspecciona metadatos para ConfigC.")
    Parser = AddCommonArguments(Parser)
    Parser.add_argument(
        "--InputTable",
        default=None,
        help="CSV de entrada. Por defecto usa Tables/DatasetFiltered.csv del experimento.",
    )
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    LogPath = Paths.LogsDirectory / "Step02InspectMetadata.log"
    Logger = CreateLogger("Step02InspectMetadata", LogPath)

    ProjectConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.ProjectConfig)
    ProjectConfig = LoadYaml(ProjectConfigPath)
    ValidateProjectConfig(ProjectConfig)

    if Args.InputTable is None:
        InputTablePath = Paths.TablesDirectory / "DatasetFiltered.csv"
    else:
        InputTablePath = ResolveProjectPath(Paths.ProjectRoot, Args.InputTable)

    if not InputTablePath.exists():
        raise FileNotFoundError(
            f"No existe la tabla para inspección de metadatos: {InputTablePath}. "
            "Ejecuta primero Scripts/Step01BuildDatasetIndex.py."
        )

    Logger.info("Leyendo tabla filtrada: %s", InputTablePath)
    DataFrame = pd.read_csv(InputTablePath)

    if DataFrame.empty:
        raise ValueError(f"La tabla está vacía: {InputTablePath}")

    MetadataRows = []

    for Column in DataFrame.columns:
        MetadataRows.append(
            {
                "Column": Column,
                "Dtype": str(DataFrame[Column].dtype),
                "NonNull": int(DataFrame[Column].notna().sum()),
                "Null": int(DataFrame[Column].isna().sum()),
            }
        )

    MetadataColumns = pd.DataFrame(MetadataRows)
    ContextMetadataCheck = BuildContextSummary(DataFrame)
    ContextCandidates = FindContextColumnCandidates(DataFrame)

    ConfigCReady = (
        ContextCandidates["WindU"] is not None
        and ContextCandidates["WindV"] is not None
        and ContextCandidates["SolarAzimuth"] is not None
    )

    OutputPaths = {
        "MetadataColumns": Paths.TablesDirectory / "MetadataColumns.csv",
        "ContextMetadataCheck": Paths.TablesDirectory / "ContextMetadataCheck.csv",
    }

    MetadataColumns.to_csv(OutputPaths["MetadataColumns"], index=False)
    ContextMetadataCheck.to_csv(OutputPaths["ContextMetadataCheck"], index=False)

    AuditPath = Paths.AuditDirectory / "InspectMetadataAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    Audit = BuildAuditRecord(
        ScriptName="Step02InspectMetadata.py",
        RunTag=Args.RunTag,
        Parameters={
            "ProjectConfig": str(ProjectConfigPath),
            "InputTable": str(InputTablePath),
        },
        Inputs={"InputTable": str(InputTablePath)},
        Outputs={Key: str(Value) for Key, Value in OutputPaths.items()} | {"Audit": str(AuditPath)},
        Status="Success",
        Details={
            "Rows": int(len(DataFrame)),
            "Columns": int(len(DataFrame.columns)),
            "ContextCandidates": ContextCandidates,
            "ConfigCReady": bool(ConfigCReady),
            "ConfigCReadyCondition": "WindU + WindV + SolarAzimuth present.",
        },
    )

    WriteJson(Audit, AuditPath)

    for Key, OutputPath in OutputPaths.items():
        RelativePath = OutputPath.relative_to(Paths.RunDirectory)
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step02InspectMetadata",
            Config="Project",
            Model="None",
            OutputType="Table",
            RelativePath=str(RelativePath),
            Created=OutputPath.exists(),
            Description=Key,
        )

    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=Args.RunTag,
        Step="Step02InspectMetadata",
        Config="Project",
        Model="None",
        OutputType="Audit",
        RelativePath=str(AuditPath.relative_to(Paths.RunDirectory)),
        Created=AuditPath.exists(),
        Description="InspectMetadataAudit",
    )

    Logger.info("Metadatos inspeccionados.")
    Logger.info("ContextCandidates: %s", ContextCandidates)
    Logger.info("ConfigCReady: %s", ConfigCReady)
    Logger.info("Audit: %s", AuditPath)


if __name__ == "__main__":
    Main()
