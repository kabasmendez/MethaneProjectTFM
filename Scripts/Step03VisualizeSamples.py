#!/usr/bin/env python3
"""
Step03VisualizeSamples.py

Visualiza muestras reales del dataset filtrado.

Entradas:
- Outputs/Experiments/<RunTag>/Tables/DatasetFiltered.csv
- Dataset TACO definido en ProjectConfig.yaml
- VisualizationConfig.yaml

Salidas:
- Figures/SampleGrid.png
- Figures/SampleOverview_<SampleId>.png
- Tables/VisualizedSamples.csv
- Audit/VisualizeSamplesAudit.json
- Logs/Step03VisualizeSamples.log
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
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.ReadTacoSample import ReadFullTacoSample
from Source.RunUtils import AddCommonArguments, ResolveProjectPath, ValidateCommonArguments
from Source.TacoIndex import GetSampleTable, LoadTacoDataset
from Source.VisualizationStyle import ApplyMatplotlibStyle, LoadVisualizationConfig
from Source.VisualizeSamples import PlotSampleGrid, PlotSampleOverview


def SelectSampleIds(
    DatasetFiltered: pd.DataFrame,
    MaxSamples: int,
    Seed: int,
    RequestedSampleIds: list[str] | None = None,
) -> list[str]:
    """Selecciona muestras para visualización."""
    if "SampleId" not in DatasetFiltered.columns:
        raise KeyError("DatasetFiltered debe contener columna SampleId.")

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


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Visualiza muestras reales del dataset filtrado.")
    Parser = AddCommonArguments(Parser)
    Parser.add_argument(
        "--VisualizationConfig",
        default="Configs/VisualizationConfig.yaml",
        help="Ruta al archivo VisualizationConfig.yaml.",
    )
    Parser.add_argument(
        "--MaxSamples",
        type=int,
        default=8,
        help="Número máximo de muestras para visualizar.",
    )
    Parser.add_argument(
        "--SampleIds",
        nargs="*",
        default=None,
        help="Lista opcional de SampleIds específicos.",
    )
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    LogPath = Paths.LogsDirectory / "Step03VisualizeSamples.log"
    Logger = CreateLogger("Step03VisualizeSamples", LogPath)

    ProjectConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.ProjectConfig)
    VisualizationConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.VisualizationConfig)

    ProjectConfig = LoadYaml(ProjectConfigPath)
    ValidateProjectConfig(ProjectConfig)

    VisualConfig = LoadVisualizationConfig(VisualizationConfigPath)
    ApplyMatplotlibStyle(VisualConfig)

    DatasetFilteredPath = Paths.TablesDirectory / "DatasetFiltered.csv"

    if not DatasetFilteredPath.exists():
        raise FileNotFoundError(
            f"No existe DatasetFiltered.csv: {DatasetFilteredPath}. Ejecuta primero Step01BuildDatasetIndex.py."
        )

    DatasetFiltered = pd.read_csv(DatasetFilteredPath)

    if DatasetFiltered.empty:
        raise ValueError(f"DatasetFiltered está vacío: {DatasetFilteredPath}")

    SampleIds = SelectSampleIds(
        DatasetFiltered,
        MaxSamples=Args.MaxSamples,
        Seed=int(ProjectConfig["Seed"]),
        RequestedSampleIds=Args.SampleIds,
    )

    Logger.info("Muestras seleccionadas: %s", SampleIds)

    Dataset, DatasetInfo = LoadTacoDataset(
        ProjectConfig["Dataset"]["DataRoot"],
        ProjectConfig["Dataset"]["DatasetName"],
    )

    SampleTable = GetSampleTable(Dataset)
    ExpectedShapes = ProjectConfig.get("ExpectedShapes", {})

    SampleDataList = []
    ValidationRows = []

    for SampleId in SampleIds:
        Logger.info("Leyendo muestra: %s", SampleId)
        SampleData = ReadFullTacoSample(
            Dataset=Dataset,
            SampleTable=SampleTable,
            SampleId=SampleId,
            ExpectedShapes=ExpectedShapes,
        )

        SampleDataList.append(SampleData)

        Validation = SampleData["Validation"].copy()
        Validation["SampleIndex"] = SampleData["SampleIndex"]
        Validation["TargetPath"] = SampleData["Paths"]["Target"]
        Validation["ReferencePath"] = SampleData["Paths"]["Reference"]
        Validation["PlumePath"] = SampleData["Paths"]["Plume"]
        Validation["CH4Path"] = SampleData["Paths"]["CH4"]
        Validation["DemPath"] = SampleData["Paths"]["Dem"]

        ValidationRows.append(Validation)

        OverviewPath = Paths.FiguresDirectory / f"SampleOverview_{SampleId}.png"
        PlotSampleOverview(
            SampleData=SampleData,
            VisualConfig=VisualConfig,
            SavePath=OverviewPath,
            ShowFigure=False,
        )

    GridPath = Paths.FiguresDirectory / "SampleGrid.png"
    PlotSampleGrid(
        SampleDataList=SampleDataList,
        VisualConfig=VisualConfig,
        SavePath=GridPath,
        ShowFigure=False,
    )

    VisualizedSamplesPath = Paths.TablesDirectory / "VisualizedSamples.csv"
    VisualizedSamples = pd.DataFrame(ValidationRows)
    VisualizedSamples.to_csv(VisualizedSamplesPath, index=False)

    AuditPath = Paths.AuditDirectory / "VisualizeSamplesAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    OutputFiles = {
        "SampleGrid": GridPath,
        "VisualizedSamples": VisualizedSamplesPath,
        "Audit": AuditPath,
    }

    for SampleId in SampleIds:
        OutputFiles[f"SampleOverview_{SampleId}"] = Paths.FiguresDirectory / f"SampleOverview_{SampleId}.png"

    Audit = BuildAuditRecord(
        ScriptName="Step03VisualizeSamples.py",
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
        Outputs={Key: str(Value) for Key, Value in OutputFiles.items()},
        Status="Success",
        Details={
            "SampleCount": len(SampleIds),
            "SelectedSampleIds": SampleIds,
            "ValidationRows": VisualizedSamples.to_dict(orient="records"),
        },
    )

    WriteJson(Audit, AuditPath)

    RegisterItems = [
        ("Figure", GridPath, "Grid de muestras con RGB target y máscara plume."),
        ("Table", VisualizedSamplesPath, "Validación de muestras visualizadas."),
        ("Audit", AuditPath, "Auditoría de visualización de muestras."),
    ]

    for SampleId in SampleIds:
        RegisterItems.append(
            (
                "Figure",
                Paths.FiguresDirectory / f"SampleOverview_{SampleId}.png",
                f"Panel completo de muestra {SampleId}.",
            )
        )

    for OutputType, OutputPath, Description in RegisterItems:
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step03VisualizeSamples",
            Config="Project",
            Model="None",
            OutputType=OutputType,
            RelativePath=str(OutputPath.relative_to(Paths.RunDirectory)),
            Created=OutputPath.exists(),
            Description=Description,
        )

    Logger.info("Step03VisualizeSamples completado.")
    Logger.info("SampleGrid: %s", GridPath)
    Logger.info("VisualizedSamples: %s", VisualizedSamplesPath)
    Logger.info("Audit: %s", AuditPath)


if __name__ == "__main__":
    Main()
