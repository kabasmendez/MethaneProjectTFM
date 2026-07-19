#!/usr/bin/env python3
"""
Step01BuildDatasetIndex.py

Construye DatasetIndex, aplica filtros obligatorios y genera splits reproducibles.

Salidas:
- Tables/DatasetIndex.csv
- Tables/ProductIndex.csv
- Tables/ProductPresence.csv
- Tables/ColumnSummary.csv
- Tables/DetectedColumns.csv
- Tables/DatasetFiltered.csv
- Tables/FilterSummary.csv
- Tables/SplitAll.csv
- Tables/SplitTrain.csv
- Tables/SplitValidation.csv
- Tables/SplitTest.csv
- Tables/SplitSummary.csv
- Audit/DatasetIndexAudit.json
- Logs/Step01BuildDatasetIndex.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml, ValidateProjectConfig
from Source.Filters import ApplyMandatoryFilters
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ResolveProjectPath, ValidateCommonArguments
from Source.SplitUtils import CreateTrainValidationTestSplit
from Source.TacoIndex import (
    BuildDatasetIndex,
    BuildProductIndex,
    BuildProductPresence,
    GetFlattenedTable,
    GetSampleTable,
    LoadTacoDataset,
    SummarizeColumns,
)


def RegisterOutput(OutputIndexPath: Path, RunTag: str, OutputType: str, PathObject: Path, Description: str):
    """Registra una salida en OutputIndex.csv."""
    RelativePath = PathObject.relative_to(PathObject.parents[2])
    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=RunTag,
        Step="Step01BuildDatasetIndex",
        Config="Project",
        Model="None",
        OutputType=OutputType,
        RelativePath=str(RelativePath),
        Created=PathObject.exists(),
        Description=Description,
    )


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Construye índice TACO filtrado y splits.")
    Parser = AddCommonArguments(Parser)
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    LogPath = Paths.LogsDirectory / "Step01BuildDatasetIndex.log"
    Logger = CreateLogger("Step01BuildDatasetIndex", LogPath)

    Logger.info("Iniciando Step01BuildDatasetIndex.")
    Logger.info("RunTag: %s", Args.RunTag)

    ProjectConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.ProjectConfig)
    ProjectConfig = LoadYaml(ProjectConfigPath)
    ValidateProjectConfig(ProjectConfig)

    DataRoot = ProjectConfig["Dataset"]["DataRoot"]
    DatasetName = ProjectConfig["Dataset"]["DatasetName"]

    Logger.info("Cargando dataset TACO: %s/%s", DataRoot, DatasetName)
    Dataset, DatasetInfo = LoadTacoDataset(DataRoot, DatasetName)

    Logger.info("Extrayendo tablas TACO.")
    SampleTable = GetSampleTable(Dataset)
    FlattenedTable = GetFlattenedTable(Dataset)

    Logger.info("Construyendo índice de productos.")
    ProductIndex = BuildProductIndex(FlattenedTable)
    ProductPresence = BuildProductPresence(ProductIndex)
    DatasetIndex = BuildDatasetIndex(SampleTable, ProductPresence)

    Logger.info("Aplicando filtros obligatorios.")
    DatasetFiltered, FilterSummary, DetectedColumns = ApplyMandatoryFilters(
        DatasetIndex,
        ProjectConfig,
    )

    Logger.info("Muestras iniciales: %d", len(DatasetIndex))
    Logger.info("Muestras filtradas: %d", len(DatasetFiltered))

    Logger.info("Creando splits.")
    SplitAll, SplitSummary = CreateTrainValidationTestSplit(DatasetFiltered, ProjectConfig)

    OutputPaths = {
        "DatasetIndex": Paths.TablesDirectory / "DatasetIndex.csv",
        "ProductIndex": Paths.TablesDirectory / "ProductIndex.csv",
        "ProductPresence": Paths.TablesDirectory / "ProductPresence.csv",
        "ColumnSummary": Paths.TablesDirectory / "ColumnSummary.csv",
        "DetectedColumns": Paths.TablesDirectory / "DetectedColumns.csv",
        "DatasetFiltered": Paths.TablesDirectory / "DatasetFiltered.csv",
        "FilterSummary": Paths.TablesDirectory / "FilterSummary.csv",
        "SplitAll": Paths.TablesDirectory / "SplitAll.csv",
        "SplitTrain": Paths.TablesDirectory / "SplitTrain.csv",
        "SplitValidation": Paths.TablesDirectory / "SplitValidation.csv",
        "SplitTest": Paths.TablesDirectory / "SplitTest.csv",
        "SplitSummary": Paths.TablesDirectory / "SplitSummary.csv",
    }

    DatasetIndex.to_csv(OutputPaths["DatasetIndex"], index=False)
    ProductIndex.to_csv(OutputPaths["ProductIndex"], index=False)
    ProductPresence.to_csv(OutputPaths["ProductPresence"], index=False)
    SummarizeColumns(DatasetIndex).to_csv(OutputPaths["ColumnSummary"], index=False)
    DetectedColumns.to_csv(OutputPaths["DetectedColumns"], index=False)
    DatasetFiltered.to_csv(OutputPaths["DatasetFiltered"], index=False)
    FilterSummary.to_csv(OutputPaths["FilterSummary"], index=False)

    SplitAll.to_csv(OutputPaths["SplitAll"], index=False)
    SplitAll.loc[SplitAll["Split"] == "Train"].to_csv(OutputPaths["SplitTrain"], index=False)
    SplitAll.loc[SplitAll["Split"] == "Validation"].to_csv(
        OutputPaths["SplitValidation"],
        index=False,
    )
    SplitAll.loc[SplitAll["Split"] == "Test"].to_csv(OutputPaths["SplitTest"], index=False)
    SplitSummary.to_csv(OutputPaths["SplitSummary"], index=False)

    AuditPath = Paths.AuditDirectory / "DatasetIndexAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    Audit = BuildAuditRecord(
        ScriptName="Step01BuildDatasetIndex.py",
        RunTag=Args.RunTag,
        Parameters={
            "ProjectConfig": str(ProjectConfigPath),
            "Seed": ProjectConfig["Seed"],
            "DatasetName": DatasetName,
        },
        Inputs={
            "Dataset": DatasetInfo,
            "ProjectConfig": str(ProjectConfigPath),
        },
        Outputs={Key: str(Value) for Key, Value in OutputPaths.items()} | {"Audit": str(AuditPath)},
        Status="Success",
        Details={
            "InitialSamples": int(len(DatasetIndex)),
            "FilteredSamples": int(len(DatasetFiltered)),
            "SplitSummary": SplitSummary.to_dict(orient="records"),
            "FilterSummary": FilterSummary.to_dict(orient="records"),
        },
    )

    WriteJson(Audit, AuditPath)

    for Key, OutputPath in OutputPaths.items():
        RegisterOutput(OutputIndexPath, Args.RunTag, "Table", OutputPath, Key)

    RegisterOutput(OutputIndexPath, Args.RunTag, "Audit", AuditPath, "DatasetIndexAudit")

    Logger.info("Step01 completado correctamente.")
    Logger.info("Audit: %s", AuditPath)


if __name__ == "__main__":
    Main()
