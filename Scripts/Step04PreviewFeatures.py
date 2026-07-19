#!/usr/bin/env python3
"""
Step04PreviewFeatures.py

Preview técnico de features ConfigA y ConfigB.

Este script:
- lee muestras reales del dataset filtrado;
- calcula ConfigA y ConfigB;
- genera figuras de inspección;
- guarda estadísticas por feature;
- audita explícitamente que MBMPPlus usa ground-truth mask.

No construye tensores finales de entrenamiento.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml, ValidateProjectConfig
from Source.FeatureEngineering import (
    CONFIG_A_FEATURES,
    CONFIG_B_FEATURES,
    BuildFeatureDictionary,
    BuildFeatureSummaryTable,
)
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.ReadTacoSample import ReadFullTacoSample
from Source.RunUtils import AddCommonArguments, ResolveProjectPath, ValidateCommonArguments
from Source.TacoIndex import GetSampleTable, LoadTacoDataset
from Source.VisualizationStyle import ApplyMatplotlibStyle, LoadVisualizationConfig
from Source.VisualizeFeatures import PlotFeaturePreview


def SelectSampleIds(
    DatasetFiltered: pd.DataFrame,
    MaxSamples: int,
    Seed: int,
    RequestedSampleIds: list[str] | None = None,
) -> list[str]:
    """Selecciona muestras para preview."""
    if "SampleId" not in DatasetFiltered.columns:
        raise KeyError("DatasetFiltered debe contener SampleId.")

    AvailableIds = DatasetFiltered["SampleId"].astype(str).tolist()

    if RequestedSampleIds:
        Missing = [SampleId for SampleId in RequestedSampleIds if SampleId not in set(AvailableIds)]
        if Missing:
            raise ValueError(f"SampleIds solicitados no existen en DatasetFiltered: {Missing}")
        return RequestedSampleIds

    Random = random.Random(Seed)

    if len(AvailableIds) <= MaxSamples:
        return AvailableIds

    return Random.sample(AvailableIds, MaxSamples)


def SaveStats(
    Rows: list[dict],
    OutputPath: Path,
) -> None:
    """Guarda tabla de estadísticas."""
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(Rows).to_csv(OutputPath, index=False)


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Preview de features ConfigA y ConfigB.")
    Parser = AddCommonArguments(Parser)
    Parser.add_argument(
        "--VisualizationConfig",
        default="Configs/VisualizationConfig.yaml",
        help="Ruta al archivo VisualizationConfig.yaml.",
    )
    Parser.add_argument("--MaxSamples", type=int, default=2)
    Parser.add_argument("--SampleIds", nargs="*", default=None)
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    LogPath = Paths.LogsDirectory / "Step04PreviewFeatures.log"
    Logger = CreateLogger("Step04PreviewFeatures", LogPath)

    ProjectConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.ProjectConfig)
    VisualizationConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.VisualizationConfig)

    ProjectConfig = LoadYaml(ProjectConfigPath)
    ValidateProjectConfig(ProjectConfig)

    VisualConfig = LoadVisualizationConfig(VisualizationConfigPath)
    ApplyMatplotlibStyle(VisualConfig)

    DatasetFilteredPath = Paths.TablesDirectory / "DatasetFiltered.csv"

    if not DatasetFilteredPath.exists():
        raise FileNotFoundError(
            f"No existe DatasetFiltered.csv: {DatasetFilteredPath}. Ejecuta Step01 primero."
        )

    DatasetFiltered = pd.read_csv(DatasetFilteredPath)

    SampleIds = SelectSampleIds(
        DatasetFiltered=DatasetFiltered,
        MaxSamples=Args.MaxSamples,
        Seed=int(ProjectConfig["Seed"]),
        RequestedSampleIds=Args.SampleIds,
    )

    Logger.info("SampleIds seleccionados: %s", SampleIds)

    Dataset, DatasetInfo = LoadTacoDataset(
        ProjectConfig["Dataset"]["DataRoot"],
        ProjectConfig["Dataset"]["DatasetName"],
    )
    SampleTable = GetSampleTable(Dataset)
    ExpectedShapes = ProjectConfig.get("ExpectedShapes", {})

    ConfigAStatsRows = []
    ConfigBStatsRows = []
    CreatedFigures = []

    for SampleId in SampleIds:
        Logger.info("Leyendo muestra: %s", SampleId)

        SampleData = ReadFullTacoSample(
            Dataset=Dataset,
            SampleTable=SampleTable,
            SampleId=SampleId,
            ExpectedShapes=ExpectedShapes,
        )

        ConfigAFeatures = BuildFeatureDictionary(
            Target=SampleData["Target"],
            Reference=SampleData["Reference"],
            PlumeMask=None,
            FeatureConfig="ConfigA",
        )

        ConfigBFeatures = BuildFeatureDictionary(
            Target=SampleData["Target"],
            Reference=SampleData["Reference"],
            PlumeMask=SampleData["Plume"],
            FeatureConfig="ConfigB",
        )

        ConfigAFigurePath = (
            Paths.RunDirectory / "ConfigA" / "Figures" / f"FeaturePreview_{SampleId}.png"
        )
        ConfigBFigurePath = (
            Paths.RunDirectory / "ConfigB" / "Figures" / f"FeaturePreview_{SampleId}.png"
        )

        PlotFeaturePreview(
            Target=SampleData["Target"],
            Reference=SampleData["Reference"],
            CH4=SampleData.get("CH4"),
            Plume=SampleData["Plume"],
            FeatureDictionary=ConfigAFeatures,
            FeatureConfig="ConfigA",
            SampleId=SampleId,
            VisualConfig=VisualConfig,
            SavePath=ConfigAFigurePath,
            ShowFigure=False,
        )

        PlotFeaturePreview(
            Target=SampleData["Target"],
            Reference=SampleData["Reference"],
            CH4=SampleData.get("CH4"),
            Plume=SampleData["Plume"],
            FeatureDictionary=ConfigBFeatures,
            FeatureConfig="ConfigB",
            SampleId=SampleId,
            VisualConfig=VisualConfig,
            SavePath=ConfigBFigurePath,
            ShowFigure=False,
        )

        CreatedFigures.extend([ConfigAFigurePath, ConfigBFigurePath])

        for Row in BuildFeatureSummaryTable(ConfigAFeatures, PlumeMask=SampleData["Plume"]):
            Row["SampleId"] = SampleId
            Row["Config"] = "ConfigA"
            ConfigAStatsRows.append(Row)

        for Row in BuildFeatureSummaryTable(ConfigBFeatures, PlumeMask=SampleData["Plume"]):
            Row["SampleId"] = SampleId
            Row["Config"] = "ConfigB"
            ConfigBStatsRows.append(Row)

    ConfigAStatsPath = Paths.RunDirectory / "ConfigA" / "Tables" / "FeaturePreviewStats.csv"
    ConfigBStatsPath = Paths.RunDirectory / "ConfigB" / "Tables" / "FeaturePreviewStats.csv"

    SaveStats(ConfigAStatsRows, ConfigAStatsPath)
    SaveStats(ConfigBStatsRows, ConfigBStatsPath)

    AuditPath = Paths.AuditDirectory / "FeaturePreviewAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    Audit = BuildAuditRecord(
        ScriptName="Step04PreviewFeatures.py",
        RunTag=Args.RunTag,
        Parameters={
            "ProjectConfig": str(ProjectConfigPath),
            "VisualizationConfig": str(VisualizationConfigPath),
            "MaxSamples": Args.MaxSamples,
            "SampleIds": SampleIds,
        },
        Inputs={
            "DatasetFiltered": str(DatasetFilteredPath),
            "Dataset": DatasetInfo,
        },
        Outputs={
            "ConfigAStats": str(ConfigAStatsPath),
            "ConfigBStats": str(ConfigBStatsPath),
            "Figures": [str(PathItem) for PathItem in CreatedFigures],
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "SampleCount": len(SampleIds),
            "ConfigAFeatures": CONFIG_A_FEATURES,
            "ConfigBFeatures": CONFIG_B_FEATURES,
            "MBMPPlusMethodology": {
                "Variant": "SupervisedBackgroundCorrection",
                "UsesGroundTruthMask": True,
                "BackgroundDefinition": "plume == 0",
                "PlumeDefinition": "plume > 0",
                "RegressionModel": "Ridge",
                "Documentation": "Docs/TechnicalDecision_MBMPPlus.md",
            },
            "ImportantNote": (
                "Step04PreviewFeatures genera preview visual y estadísticas. "
                "No construye tensores finales de entrenamiento."
            ),
        },
    )

    WriteJson(Audit, AuditPath)

    RegisterItems = [
        ("Table", ConfigAStatsPath, "Estadísticas preview ConfigA."),
        ("Table", ConfigBStatsPath, "Estadísticas preview ConfigB."),
        ("Audit", AuditPath, "Auditoría preview features."),
    ]

    for FigurePath in CreatedFigures:
        ConfigName = "ConfigA" if "/ConfigA/" in str(FigurePath) else "ConfigB"
        RegisterItems.append(( "Figure", FigurePath, f"Preview visual {ConfigName}: {FigurePath.name}." ))

    for OutputType, OutputPath, Description in RegisterItems:
        ConfigName = "Project"
        if "/ConfigA/" in str(OutputPath):
            ConfigName = "ConfigA"
        elif "/ConfigB/" in str(OutputPath):
            ConfigName = "ConfigB"

        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step04PreviewFeatures",
            Config=ConfigName,
            Model="None",
            OutputType=OutputType,
            RelativePath=str(OutputPath.relative_to(Paths.RunDirectory)),
            Created=OutputPath.exists(),
            Description=Description,
        )

    Logger.info("Step04PreviewFeatures completado.")
    Logger.info("ConfigAStats: %s", ConfigAStatsPath)
    Logger.info("ConfigBStats: %s", ConfigBStatsPath)
    Logger.info("Audit: %s", AuditPath)


if __name__ == "__main__":
    Main()
