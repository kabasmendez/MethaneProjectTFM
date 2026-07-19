#!/usr/bin/env python3
"""
Step12SelectVisualizationCases.py

Selecciona casos visuales para predicciones y comparaciones.

Objetivo:
- Crear un set fijo de casos visuales compartido por todos los modelos.
- Crear casos específicos por modelo: mejores, peores, más FP, más FN.

Entradas:
- Tables/SplitTestFeatureReady.csv
- <FeatureConfig>/<ModelRunId>/Metrics/TestMetricsBySample.csv

Salidas:
- Tables/VisualizationCaseSet.csv
- <FeatureConfig>/<ModelRunId>/Tables/VisualizationCaseSet_<ModelRunId>.csv
- <FeatureConfig>/<ModelRunId>/Audit/VisualizationCaseSelectionAudit.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


def BuildModelRunId(ModelName: str, RunName: str | None) -> str:
    if RunName is None or str(RunName).strip() == "":
        return ModelName
    return f"{ModelName}_{str(RunName).strip().replace(' ', '')}"


def LoadTestSplit(RunDirectory: Path) -> pd.DataFrame:
    PathItem = RunDirectory / "Tables" / "SplitTestFeatureReady.csv"
    if not PathItem.exists():
        raise FileNotFoundError(PathItem)

    Table = pd.read_csv(PathItem)
    if "SampleId" not in Table.columns:
        raise KeyError(f"{PathItem} debe contener SampleId.")

    Table = Table.copy()
    Table["SampleId"] = Table["SampleId"].astype(str)
    return Table


def LoadMetrics(ModelRoot: Path) -> pd.DataFrame:
    PathItem = ModelRoot / "Metrics" / "TestMetricsBySample.csv"
    if not PathItem.exists():
        raise FileNotFoundError(
            f"No existe {PathItem}. Ejecuta primero Step11EvaluateSegmentationModel.py."
        )

    Table = pd.read_csv(PathItem)
    Required = [
        "SampleId",
        "Dice",
        "IoU",
        "GroundTruthPixels",
        "PredictedPixels",
        "FalsePositivePixels",
        "FalseNegativePixels",
    ]
    Missing = [Column for Column in Required if Column not in Table.columns]
    if Missing:
        raise KeyError(f"Faltan columnas en {PathItem}: {Missing}")

    Table = Table.copy()
    Table["SampleId"] = Table["SampleId"].astype(str)
    return Table


def CreateFixedCases(
    TestSplit: pd.DataFrame,
    Metrics: pd.DataFrame,
    Count: int,
    Seed: int,
) -> pd.DataFrame:
    """
    Crea casos fijos reproducibles.

    Estrategia:
    - Si hay métricas por muestra, estratifica por tamaño de pluma.
    - Selecciona casos de pluma pequeña, media y grande.
    - Si faltan casos, completa con muestreo aleatorio reproducible.
    """
    Joined = TestSplit[["SampleId"]].merge(
        Metrics[["SampleId", "GroundTruthPixels"]],
        on="SampleId",
        how="left",
    )

    Joined["GroundTruthPixels"] = Joined["GroundTruthPixels"].fillna(0)

    Positive = Joined[Joined["GroundTruthPixels"] > 0].copy()

    Rows = []

    if len(Positive) >= 3:
        Positive["PlumeSizeBin"] = pd.qcut(
            Positive["GroundTruthPixels"],
            q=3,
            labels=["SmallPlume", "MediumPlume", "LargePlume"],
            duplicates="drop",
        )

        PerBin = max(1, Count // max(1, Positive["PlumeSizeBin"].nunique()))

        for BinName, Group in Positive.groupby("PlumeSizeBin", observed=False):
            Sampled = Group.sample(
                n=min(PerBin, len(Group)),
                random_state=Seed,
            )
            for _, Row in Sampled.iterrows():
                Rows.append(
                    {
                        "CaseGroup": "FixedComparisonCases",
                        "SampleId": Row["SampleId"],
                        "Split": "Test",
                        "SelectionReason": str(BinName),
                        "GroundTruthPixels": int(Row["GroundTruthPixels"]),
                    }
                )

    ExistingIds = {Row["SampleId"] for Row in Rows}

    if len(Rows) < Count:
        Remaining = Joined[~Joined["SampleId"].isin(ExistingIds)].copy()
        if not Remaining.empty:
            Sampled = Remaining.sample(
                n=min(Count - len(Rows), len(Remaining)),
                random_state=Seed,
            )
            for _, Row in Sampled.iterrows():
                Rows.append(
                    {
                        "CaseGroup": "FixedComparisonCases",
                        "SampleId": Row["SampleId"],
                        "Split": "Test",
                        "SelectionReason": "RandomReproducibleFill",
                        "GroundTruthPixels": int(Row["GroundTruthPixels"]),
                    }
                )

    Output = pd.DataFrame(Rows).head(Count).copy()
    Output["Order"] = range(1, len(Output) + 1)

    return Output


def SelectMetricCases(
    Metrics: pd.DataFrame,
    GroupName: str,
    SortColumn: str,
    Ascending: bool,
    Count: int,
    SelectionReason: str,
) -> pd.DataFrame:
    """Selecciona casos por métrica."""
    Table = Metrics.sort_values(SortColumn, ascending=Ascending).head(Count).copy()

    Output = pd.DataFrame(
        {
            "CaseGroup": GroupName,
            "SampleId": Table["SampleId"].astype(str),
            "Split": Table["Split"] if "Split" in Table.columns else "Test",
            "SelectionReason": SelectionReason,
            "GroundTruthPixels": Table["GroundTruthPixels"].astype(int),
            "Dice": Table["Dice"].astype(float),
            "IoU": Table["IoU"].astype(float),
            "FalsePositivePixels": Table["FalsePositivePixels"].astype(int),
            "FalseNegativePixels": Table["FalseNegativePixels"].astype(int),
        }
    )

    Output["Order"] = range(1, len(Output) + 1)
    return Output


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Selecciona casos visuales.")
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--FeatureConfig", required=True, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--ModelName", required=True)
    Parser.add_argument("--RunName", required=True)

    Parser.add_argument("--FixedCount", type=int, default=12)
    Parser.add_argument("--BestCount", type=int, default=6)
    Parser.add_argument("--WorstCount", type=int, default=6)
    Parser.add_argument("--ErrorCount", type=int, default=6)
    Parser.add_argument("--Seed", type=int, default=42)

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    ModelRunId = BuildModelRunId(Args.ModelName, Args.RunName)
    ModelRoot = Paths.RunDirectory / Args.FeatureConfig / ModelRunId

    TableDirectory = ModelRoot / "Tables"
    AuditDirectory = ModelRoot / "Audit"
    TableDirectory.mkdir(parents=True, exist_ok=True)
    AuditDirectory.mkdir(parents=True, exist_ok=True)

    Logger = CreateLogger(
        f"Step12SelectVisualizationCases_{Args.FeatureConfig}_{ModelRunId}",
        Paths.LogsDirectory / f"Step12SelectVisualizationCases_{Args.FeatureConfig}_{ModelRunId}.log",
    )

    TestSplit = LoadTestSplit(Paths.RunDirectory)
    Metrics = LoadMetrics(ModelRoot)

    FixedPath = Paths.RunDirectory / "Tables" / "VisualizationCaseSet.csv"

    if FixedPath.exists():
        FixedCases = pd.read_csv(FixedPath)
        FixedCases["SampleId"] = FixedCases["SampleId"].astype(str)
        Logger.info("Using existing fixed visualization case set: %s", FixedPath)
    else:
        FixedCases = CreateFixedCases(
            TestSplit=TestSplit,
            Metrics=Metrics,
            Count=Args.FixedCount,
            Seed=Args.Seed,
        )
        FixedCases.to_csv(FixedPath, index=False)
        Logger.info("Created fixed visualization case set: %s", FixedPath)

    BestCases = SelectMetricCases(
        Metrics=Metrics,
        GroupName="BestPredictions",
        SortColumn="Dice",
        Ascending=False,
        Count=Args.BestCount,
        SelectionReason="HighestDice",
    )

    WorstCases = SelectMetricCases(
        Metrics=Metrics,
        GroupName="WorstPredictions",
        SortColumn="Dice",
        Ascending=True,
        Count=Args.WorstCount,
        SelectionReason="LowestDice",
    )

    FpCases = SelectMetricCases(
        Metrics=Metrics,
        GroupName="HighFalsePositive",
        SortColumn="FalsePositivePixels",
        Ascending=False,
        Count=Args.ErrorCount,
        SelectionReason="HighestFalsePositivePixels",
    )

    FnCases = SelectMetricCases(
        Metrics=Metrics,
        GroupName="HighFalseNegative",
        SortColumn="FalseNegativePixels",
        Ascending=False,
        Count=Args.ErrorCount,
        SelectionReason="HighestFalseNegativePixels",
    )

    ModelCases = pd.concat(
        [FixedCases, BestCases, WorstCases, FpCases, FnCases],
        ignore_index=True,
    )

    ModelCases["RunTag"] = Args.RunTag
    ModelCases["FeatureConfig"] = Args.FeatureConfig
    ModelCases["ModelName"] = Args.ModelName
    ModelCases["RunName"] = Args.RunName
    ModelCases["ModelRunId"] = ModelRunId

    ModelCasePath = TableDirectory / f"VisualizationCaseSet_{ModelRunId}.csv"
    AuditPath = AuditDirectory / "VisualizationCaseSelectionAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    ModelCases.to_csv(ModelCasePath, index=False)

    Audit = BuildAuditRecord(
        ScriptName="Step12SelectVisualizationCases.py",
        RunTag=Args.RunTag,
        Parameters=vars(Args),
        Inputs={
            "TestSplit": str(Paths.RunDirectory / "Tables" / "SplitTestFeatureReady.csv"),
            "TestMetricsBySample": str(ModelRoot / "Metrics" / "TestMetricsBySample.csv"),
        },
        Outputs={
            "FixedVisualizationCaseSet": str(FixedPath),
            "ModelVisualizationCaseSet": str(ModelCasePath),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "ModelRunId": ModelRunId,
            "FixedCases": int(len(FixedCases)),
            "ModelCases": int(len(ModelCases)),
            "CaseGroups": sorted(ModelCases["CaseGroup"].dropna().unique().tolist()),
        },
    )

    WriteJson(Audit, AuditPath)

    for OutputType, OutputPath, Description in [
        ("Table", FixedPath, "Casos fijos de visualización compartidos."),
        ("Table", ModelCasePath, f"Casos visuales para {ModelRunId}."),
        ("Audit", AuditPath, f"Auditoría de selección de casos visuales para {ModelRunId}."),
    ]:
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step12SelectVisualizationCases",
            Config=Args.FeatureConfig,
            Model=ModelRunId,
            OutputType=OutputType,
            RelativePath=str(OutputPath.relative_to(Paths.RunDirectory)),
            Created=OutputPath.exists(),
            Description=Description,
        )

    print("\n=== Visualization cases selected ===")
    print("Fixed cases:", FixedPath)
    print("Model cases:", ModelCasePath)
    print(ModelCases[["CaseGroup", "Order", "SampleId", "SelectionReason"]].head(30).to_string(index=False))


if __name__ == "__main__":
    Main()
