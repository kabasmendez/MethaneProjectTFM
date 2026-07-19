#!/usr/bin/env python3
"""
Step15CompareExperiments.py

Compara ejecuciones ya evaluadas.

Formato de Items:
  RunTag:FeatureConfig:ModelRunId

Ejemplo:
  Exp261944:ConfigB:TransformerUNet_Bs4Ep10
  Exp261944:ConfigB:EnhancedUNet_Ep5Preview

Salidas:
- Outputs/Comparisons/<ComparisonTag>/Tables/ComparisonSummary.csv
- Outputs/Comparisons/<ComparisonTag>/Tables/ComparisonBySample.csv
- Outputs/Comparisons/<ComparisonTag>/Figures/MetricBars.png
- Outputs/Comparisons/<ComparisonTag>/Figures/DiceScatter.png
- Outputs/Comparisons/<ComparisonTag>/Figures/DiceDifferenceHistogram.png
- Outputs/Comparisons/<ComparisonTag>/Audit/CompareExperimentsAudit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import BuildAuditRecord, WriteJson
from Source.VisualizationStyle import ApplyMatplotlibStyle, LoadVisualizationConfig


def ParseItem(Item: str) -> dict[str, str]:
    Parts = Item.split(":")
    if len(Parts) != 3:
        raise ValueError(
            f"Item inválido: {Item}. Formato esperado RunTag:FeatureConfig:ModelRunId"
        )
    return {
        "RunTag": Parts[0],
        "FeatureConfig": Parts[1],
        "ModelRunId": Parts[2],
        "ItemId": Item,
    }


def GetModelRoot(ProjectRoot: Path, Item: dict[str, str]) -> Path:
    return (
        ProjectRoot
        / "Outputs"
        / "Experiments"
        / Item["RunTag"]
        / Item["FeatureConfig"]
        / Item["ModelRunId"]
    )


def LoadRunTables(ProjectRoot: Path, Item: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ModelRoot = GetModelRoot(ProjectRoot, Item)

    SummaryPath = ModelRoot / "Metrics" / "TestMetricsSummary.csv"
    BySamplePath = ModelRoot / "Metrics" / "TestMetricsBySample.csv"

    if not SummaryPath.exists():
        raise FileNotFoundError(f"No existe summary: {SummaryPath}")

    if not BySamplePath.exists():
        raise FileNotFoundError(f"No existe by-sample: {BySamplePath}")

    Summary = pd.read_csv(SummaryPath)
    BySample = pd.read_csv(BySamplePath)

    Summary["ItemId"] = Item["ItemId"]
    Summary["RunTag"] = Item["RunTag"]
    Summary["FeatureConfig"] = Item["FeatureConfig"]
    Summary["ModelRunId"] = Item["ModelRunId"]

    BySample["ItemId"] = Item["ItemId"]
    BySample["RunTag"] = Item["RunTag"]
    BySample["FeatureConfig"] = Item["FeatureConfig"]
    BySample["ModelRunId"] = Item["ModelRunId"]

    return Summary, BySample


def SaveMetricBars(Summary: pd.DataFrame, OutputPath: Path) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)

    Metrics = [Metric for Metric in ["MeanDice", "MeanIoU", "GlobalDice", "GlobalIoU"] if Metric in Summary.columns]

    Figure, Axis = plt.subplots(figsize=(11, 5.5))

    PlotTable = Summary.set_index("ModelRunId")[Metrics]
    PlotTable.plot(kind="bar", ax=Axis)

    Axis.set_ylim(0, 1)
    Axis.set_title("Comparison of segmentation metrics")
    Axis.set_xlabel("")
    Axis.set_ylabel("Score")
    Axis.grid(True, axis="y", alpha=0.25)
    Axis.legend(loc="best")

    Figure.tight_layout()
    Figure.savefig(OutputPath, dpi=220, bbox_inches="tight")
    plt.close(Figure)


def SaveDiceScatter(BySampleWide: pd.DataFrame, Items: list[str], OutputPath: Path) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)

    Figure, Axis = plt.subplots(figsize=(6.5, 6.5))

    if len(Items) >= 2:
        X = f"Dice__{Items[0]}"
        Y = f"Dice__{Items[1]}"

        if X in BySampleWide.columns and Y in BySampleWide.columns:
            Axis.scatter(BySampleWide[X], BySampleWide[Y], alpha=0.55, s=18)
            Axis.plot([0, 1], [0, 1], linestyle="--")
            Axis.set_xlabel(Items[0])
            Axis.set_ylabel(Items[1])
            Axis.set_title("Dice by sample: model A vs model B")
            Axis.set_xlim(0, 1)
            Axis.set_ylim(0, 1)
            Axis.grid(True, alpha=0.25)

    Figure.tight_layout()
    Figure.savefig(OutputPath, dpi=220, bbox_inches="tight")
    plt.close(Figure)


def SaveDiceDifferenceHistogram(BySampleWide: pd.DataFrame, Items: list[str], OutputPath: Path) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)

    Figure, Axis = plt.subplots(figsize=(9, 5))

    if len(Items) >= 2:
        X = f"Dice__{Items[0]}"
        Y = f"Dice__{Items[1]}"

        if X in BySampleWide.columns and Y in BySampleWide.columns:
            Difference = BySampleWide[Y] - BySampleWide[X]
            Axis.hist(Difference.dropna(), bins=30)
            Axis.axvline(0, linestyle="--")
            Axis.set_title(f"Dice difference: {Items[1]} - {Items[0]}")
            Axis.set_xlabel("Dice difference")
            Axis.set_ylabel("Samples")
            Axis.grid(True, axis="y", alpha=0.25)

    Figure.tight_layout()
    Figure.savefig(OutputPath, dpi=220, bbox_inches="tight")
    plt.close(Figure)


def BuildWideBySample(BySample: pd.DataFrame) -> pd.DataFrame:
    Metrics = [
        Metric
        for Metric in [
            "Dice",
            "IoU",
            "Precision",
            "Recall",
            "FalsePositivePixels",
            "FalseNegativePixels",
            "GroundTruthPixels",
            "PredictedPixels",
        ]
        if Metric in BySample.columns
    ]

    WideParts = []

    for Metric in Metrics:
        Pivot = BySample.pivot_table(
            index="SampleId",
            columns="ItemId",
            values=Metric,
            aggfunc="first",
        )
        Pivot.columns = [f"{Metric}__{Column}" for Column in Pivot.columns]
        WideParts.append(Pivot)

    Wide = pd.concat(WideParts, axis=1).reset_index()

    return Wide


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Compara experimentos evaluados.")
    Parser.add_argument("--ComparisonTag", required=True)
    Parser.add_argument("--Items", nargs="+", required=True)
    Parser.add_argument("--VisualizationConfig", default="Configs/VisualizationConfig.yaml")
    Args = Parser.parse_args()

    ProjectRoot = Path.cwd()

    VisualConfig = LoadVisualizationConfig(ProjectRoot / Args.VisualizationConfig)
    ApplyMatplotlibStyle(VisualConfig)

    Items = [ParseItem(Item) for Item in Args.Items]

    OutputRoot = ProjectRoot / "Outputs" / "Comparisons" / Args.ComparisonTag
    TablesDir = OutputRoot / "Tables"
    FiguresDir = OutputRoot / "Figures"
    AuditDir = OutputRoot / "Audit"

    TablesDir.mkdir(parents=True, exist_ok=True)
    FiguresDir.mkdir(parents=True, exist_ok=True)
    AuditDir.mkdir(parents=True, exist_ok=True)

    Summaries = []
    BySamples = []

    for Item in Items:
        Summary, BySample = LoadRunTables(ProjectRoot, Item)
        Summaries.append(Summary)
        BySamples.append(BySample)

    ComparisonSummary = pd.concat(Summaries, ignore_index=True)
    AllBySample = pd.concat(BySamples, ignore_index=True)
    WideBySample = BuildWideBySample(AllBySample)

    SummaryPath = TablesDir / "ComparisonSummary.csv"
    BySamplePath = TablesDir / "ComparisonBySample.csv"
    AllBySamplePath = TablesDir / "AllMetricsBySampleLong.csv"

    ComparisonSummary.to_csv(SummaryPath, index=False)
    WideBySample.to_csv(BySamplePath, index=False)
    AllBySample.to_csv(AllBySamplePath, index=False)

    ItemIds = [Item["ItemId"] for Item in Items]

    MetricBarsPath = FiguresDir / "MetricBars.png"
    DiceScatterPath = FiguresDir / "DiceScatter.png"
    DiceDifferencePath = FiguresDir / "DiceDifferenceHistogram.png"

    SaveMetricBars(ComparisonSummary, MetricBarsPath)
    SaveDiceScatter(WideBySample, ItemIds, DiceScatterPath)
    SaveDiceDifferenceHistogram(WideBySample, ItemIds, DiceDifferencePath)

    AuditPath = AuditDir / "CompareExperimentsAudit.json"

    Audit = BuildAuditRecord(
        ScriptName="Step15CompareExperiments.py",
        RunTag="Comparison",
        Parameters=vars(Args),
        Inputs={
            "Items": Args.Items,
        },
        Outputs={
            "ComparisonSummary": str(SummaryPath),
            "ComparisonBySample": str(BySamplePath),
            "AllMetricsBySampleLong": str(AllBySamplePath),
            "MetricBars": str(MetricBarsPath),
            "DiceScatter": str(DiceScatterPath),
            "DiceDifferenceHistogram": str(DiceDifferencePath),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "ComparisonTag": Args.ComparisonTag,
            "Items": Args.Items,
            "ComparedSamples": int(len(WideBySample)),
        },
    )

    WriteJson(Audit, AuditPath)

    print("\n=== Comparison completed ===")
    print("ComparisonTag:", Args.ComparisonTag)
    print("Summary:", SummaryPath)
    print("BySample:", BySamplePath)
    print("Figures:", FiguresDir)
    print(ComparisonSummary[["ItemId", "MeanDice", "MeanIoU", "GlobalDice", "GlobalIoU"]].to_string(index=False))


if __name__ == "__main__":
    Main()
