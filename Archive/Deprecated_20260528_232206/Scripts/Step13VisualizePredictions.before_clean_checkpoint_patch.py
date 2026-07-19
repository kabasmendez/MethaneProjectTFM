#!/usr/bin/env python3
"""
Step13VisualizePredictions.py

Genera figuras de predicción para casos seleccionados.

Usa:
- VisualizationCaseSet_<ModelRunId>.csv
- BestModel.pt o LastModel.pt
- TestFeatures.npy / TestMasks.npy
- Dataset TACO para visualizar Target RGB, Target SWIR y CH4

Salidas:
- Figures/Predictions/*.png
- Tables/PredictionFigureIndex.csv
- Audit/VisualizePredictionsAudit.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml, ValidateProjectConfig
from Source.FeatureTensorDataset import FeatureTensorDataset
from Source.LoggingUtils import CreateLogger
from Source.Models.ModelFactory import CreateModel
from Source.Paths import CreateExperimentDirectories
from Source.ReadTacoSample import ReadFullTacoSample
from Source.RunUtils import AddCommonArguments, ResolveProjectPath, ValidateCommonArguments
from Source.TacoIndex import GetSampleTable, LoadTacoDataset
from Source.VisualizationStyle import ApplyMatplotlibStyle, LoadVisualizationConfig
from Source.VisualizeSamples import BuildRgbFromSentinel, BuildSwirComposite


def BuildModelRunId(ModelName: str, RunName: str | None) -> str:
    if RunName is None or str(RunName).strip() == "":
        return ModelName
    return f"{ModelName}_{str(RunName).strip().replace(' ', '')}"


def ResolveDevice(DeviceArgument: str) -> torch.device:
    if DeviceArgument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Device = torch.device(DeviceArgument)
    if Device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero CUDA no está disponible.")
    return Device


def LoadFeatureConfig(ProjectRoot: Path, FeatureConfig: str) -> dict[str, Any]:
    ConfigPath = ProjectRoot / "Configs" / f"{FeatureConfig}.yaml"
    if not ConfigPath.exists():
        raise FileNotFoundError(ConfigPath)
    return LoadYaml(ConfigPath)


def LoadSplitTable(RunDirectory: Path, Split: str) -> pd.DataFrame:
    PathItem = RunDirectory / "Tables" / f"Split{Split}FeatureReady.csv"
    if not PathItem.exists():
        raise FileNotFoundError(PathItem)
    Table = pd.read_csv(PathItem)
    Table["SampleId"] = Table["SampleId"].astype(str)
    return Table


def ComputeErrorMap(GroundTruth: np.ndarray, Prediction: np.ndarray) -> np.ndarray:
    GroundTruth = GroundTruth.astype(bool)
    Prediction = Prediction.astype(bool)

    ErrorMap = np.zeros(GroundTruth.shape, dtype=np.uint8)
    ErrorMap[GroundTruth & Prediction] = 1
    ErrorMap[~GroundTruth & Prediction] = 2
    ErrorMap[GroundTruth & ~Prediction] = 3

    return ErrorMap


def DrawContour(Axis, Mask: np.ndarray, Color: str, Label: str | None = None, LineWidth: float = 1.8) -> None:
    Mask = np.asarray(Mask).astype(float)
    if Mask.sum() == 0:
        return
    Axis.contour(Mask, levels=[0.5], colors=[Color], linewidths=LineWidth)
    if Label:
        Axis.plot([], [], color=Color, linewidth=LineWidth, label=Label)


def ComputeMetrics(GroundTruth: np.ndarray, Prediction: np.ndarray) -> dict[str, Any]:
    GT = GroundTruth.astype(bool).reshape(-1)
    PR = Prediction.astype(bool).reshape(-1)

    TP = int((GT & PR).sum())
    FP = int((~GT & PR).sum())
    FN = int((GT & ~PR).sum())
    TN = int((~GT & ~PR).sum())

    Eps = 1e-7
    Dice = (2 * TP + Eps) / (2 * TP + FP + FN + Eps)
    IoU = (TP + Eps) / (TP + FP + FN + Eps)
    Precision = (TP + Eps) / (TP + FP + Eps)
    Recall = (TP + Eps) / (TP + FN + Eps)

    return {
        "Dice": float(Dice),
        "IoU": float(IoU),
        "Precision": float(Precision),
        "Recall": float(Recall),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }


def SavePredictionFigure(
    OutputPath: Path,
    SampleId: str,
    CaseGroup: str,
    TargetRgb: np.ndarray,
    TargetSwir: np.ndarray,
    CH4: np.ndarray | None,
    GroundTruth: np.ndarray,
    Probability: np.ndarray,
    Prediction: np.ndarray,
    Metrics: dict[str, Any],
    VisualConfig: dict[str, Any],
) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)

    ErrorMap = ComputeErrorMap(GroundTruth, Prediction)
    ErrorCmap = ListedColormap(["#000000", "#2ca02c", "#ff7f0e", "#d62728"])

    Figure, Axes = plt.subplots(2, 3, figsize=(15, 9))
    Axes = Axes.ravel()

    Axes[0].imshow(TargetRgb)
    Axes[0].set_title("Target RGB")
    DrawContour(Axes[0], GroundTruth, "white", "GT", 2.0)
    DrawContour(Axes[0], Prediction, "#ff7f0e", "Prediction", 1.5)

    Axes[1].imshow(TargetSwir)
    Axes[1].set_title("Target SWIR")
    DrawContour(Axes[1], GroundTruth, "white", "GT", 2.0)
    DrawContour(Axes[1], Prediction, "#ff7f0e", "Prediction", 1.5)

    if CH4 is not None:
        Im = Axes[2].imshow(CH4, cmap="plasma")
        Axes[2].set_title("CH4")
        Figure.colorbar(Im, ax=Axes[2], fraction=0.046, pad=0.04)
    else:
        Axes[2].imshow(GroundTruth, cmap="gray", vmin=0, vmax=1)
        Axes[2].set_title("CH4 unavailable")
    DrawContour(Axes[2], GroundTruth, "white", "GT", 2.0)

    Axes[3].imshow(GroundTruth, cmap="gray", vmin=0, vmax=1)
    Axes[3].set_title("Ground truth")

    Im4 = Axes[4].imshow(Probability, cmap="viridis", vmin=0, vmax=1)
    Axes[4].set_title("Predicted probability")
    Figure.colorbar(Im4, ax=Axes[4], fraction=0.046, pad=0.04)

    Axes[5].imshow(ErrorMap, cmap=ErrorCmap, vmin=0, vmax=3)
    Axes[5].set_title("Error map: TP/FP/FN")

    for Axis in Axes:
        Axis.set_xticks([])
        Axis.set_yticks([])

    Handles, Labels = Axes[0].get_legend_handles_labels()
    if Handles:
        Figure.legend(Handles, Labels, loc="lower center", ncol=2)

    Title = (
        f"{CaseGroup} | SampleId: {SampleId}\n"
        f"Dice={Metrics['Dice']:.4f} | IoU={Metrics['IoU']:.4f} | "
        f"Precision={Metrics['Precision']:.4f} | Recall={Metrics['Recall']:.4f}"
    )
    Figure.suptitle(Title)
    Figure.tight_layout(rect=[0, 0.05, 1, 0.93])

    Figure.savefig(
        OutputPath,
        dpi=VisualConfig["Visualization"].get("Dpi", 300),
        bbox_inches="tight",
        facecolor=VisualConfig["Visualization"].get("FigureFaceColor", "white"),
    )
    plt.close(Figure)


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Visualiza predicciones por casos seleccionados.")
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--FeatureConfig", required=True, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--ModelName", required=True)
    Parser.add_argument("--RunName", required=True)
    Parser.add_argument("--Split", default="Test", choices=["Train", "Validation", "Test"])
    Parser.add_argument("--Checkpoint", default="BestModel.pt", choices=["BestModel.pt", "LastModel.pt"])
    Parser.add_argument("--Threshold", type=float, default=0.5)
    Parser.add_argument("--Device", default="auto")
    Parser.add_argument("--CaseGroups", nargs="+", default=["FixedComparisonCases", "BestPredictions", "WorstPredictions"])
    Parser.add_argument("--MaxPerGroup", type=int, default=12)
    Parser.add_argument("--VisualizationConfig", default="Configs/VisualizationConfig.yaml")

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    Device = ResolveDevice(Args.Device)
    ModelRunId = BuildModelRunId(Args.ModelName, Args.RunName)
    ModelRoot = Paths.RunDirectory / Args.FeatureConfig / ModelRunId

    FigureDirectory = ModelRoot / "Figures" / "Predictions"
    TableDirectory = ModelRoot / "Tables"
    AuditDirectory = ModelRoot / "Audit"

    FigureDirectory.mkdir(parents=True, exist_ok=True)
    TableDirectory.mkdir(parents=True, exist_ok=True)
    AuditDirectory.mkdir(parents=True, exist_ok=True)

    Logger = CreateLogger(
        f"Step13VisualizePredictions_{Args.FeatureConfig}_{ModelRunId}",
        Paths.LogsDirectory / f"Step13VisualizePredictions_{Args.FeatureConfig}_{ModelRunId}.log",
    )

    ProjectConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.ProjectConfig)
    ProjectConfig = LoadYaml(ProjectConfigPath)
    ValidateProjectConfig(ProjectConfig)

    VisualConfig = LoadVisualizationConfig(Paths.ProjectRoot / Args.VisualizationConfig)
    ApplyMatplotlibStyle(VisualConfig)

    FeatureConfigYaml = LoadFeatureConfig(Paths.ProjectRoot, Args.FeatureConfig)
    InputChannels = int(FeatureConfigYaml["InputChannels"])

    SplitTable = LoadSplitTable(Paths.RunDirectory, Args.Split)
    SampleIdToIndex = {
        SampleId: Index
        for Index, SampleId in enumerate(SplitTable["SampleId"].astype(str).tolist())
    }

    CasePath = TableDirectory / f"VisualizationCaseSet_{ModelRunId}.csv"
    if not CasePath.exists():
        raise FileNotFoundError(
            f"No existe {CasePath}. Ejecuta primero Step12SelectVisualizationCases.py."
        )

    Cases = pd.read_csv(CasePath)
    Cases["SampleId"] = Cases["SampleId"].astype(str)

    SelectedCases = Cases[Cases["CaseGroup"].isin(Args.CaseGroups)].copy()
    SelectedCases = (
        SelectedCases
        .sort_values(["CaseGroup", "Order"])
        .groupby("CaseGroup", group_keys=False)
        .head(Args.MaxPerGroup)
    )

    if SelectedCases.empty:
        raise ValueError("No hay casos seleccionados para visualizar.")

    FeaturePath = Paths.RunDirectory / Args.FeatureConfig / "Features" / f"{Args.Split}Features.npy"
    MaskPath = Paths.RunDirectory / Args.FeatureConfig / "Features" / f"{Args.Split}Masks.npy"

    DatasetObject = FeatureTensorDataset(
        FeaturePath=FeaturePath,
        MaskPath=MaskPath,
        ExpectedChannels=InputChannels,
        ExpectedHeight=200,
        ExpectedWidth=200,
    )

    CheckpointPath = ModelRoot / "Checkpoints" / Args.Checkpoint
    if not CheckpointPath.exists():
        raise FileNotFoundError(CheckpointPath)

    Checkpoint = torch.load(CheckpointPath, map_location=Device)
    ModelParameters = Checkpoint.get("ModelParameters", {})

    Model = CreateModel(
        ModelName=Checkpoint.get("ModelName", Args.ModelName),
        InputChannels=InputChannels,
        OutputChannels=1,
        ModelParameters=ModelParameters,
    ).to(Device)

    Model.load_state_dict(Checkpoint["ModelStateDict"])
    Model.eval()

    TacoDataset, _ = LoadTacoDataset(
        ProjectConfig["Dataset"]["DataRoot"],
        ProjectConfig["Dataset"]["DatasetName"],
    )
    SampleTable = GetSampleTable(TacoDataset)
    ExpectedShapes = ProjectConfig.get("ExpectedShapes", {})

    Rows = []

    with torch.no_grad():
        for _, Case in SelectedCases.iterrows():
            SampleId = str(Case["SampleId"])
            CaseGroup = str(Case["CaseGroup"])

            if SampleId not in SampleIdToIndex:
                Logger.warning("SampleId no encontrado en split %s: %s", Args.Split, SampleId)
                continue

            DatasetIndex = SampleIdToIndex[SampleId]
            TensorItem = DatasetObject[DatasetIndex]

            Features = TensorItem["features"].unsqueeze(0).to(Device)
            GroundTruth = TensorItem["mask"].squeeze(0).cpu().numpy().astype(np.uint8)

            Logits = Model(Features)
            Probability = torch.sigmoid(Logits)[0, 0].detach().cpu().numpy()
            Prediction = (Probability >= Args.Threshold).astype(np.uint8)

            Metrics = ComputeMetrics(GroundTruth, Prediction)

            RawSample = ReadFullTacoSample(
                Dataset=TacoDataset,
                SampleTable=SampleTable,
                SampleId=SampleId,
                ExpectedShapes=ExpectedShapes,
            )

            TargetRgb = BuildRgbFromSentinel(RawSample["Target"], VisualConfig)
            TargetSwir = BuildSwirComposite(RawSample["Target"], VisualConfig)
            CH4 = RawSample.get("CH4")

            SafeSampleId = SampleId.replace("/", "_")
            OutputPath = FigureDirectory / f"{CaseGroup}_{int(Case['Order']):03d}_{SafeSampleId}.png"

            SavePredictionFigure(
                OutputPath=OutputPath,
                SampleId=SampleId,
                CaseGroup=CaseGroup,
                TargetRgb=TargetRgb,
                TargetSwir=TargetSwir,
                CH4=CH4,
                GroundTruth=GroundTruth,
                Probability=Probability,
                Prediction=Prediction,
                Metrics=Metrics,
                VisualConfig=VisualConfig,
            )

            Rows.append(
                {
                    "CaseGroup": CaseGroup,
                    "Order": int(Case["Order"]),
                    "SampleId": SampleId,
                    "DatasetIndex": int(DatasetIndex),
                    "FigurePath": str(OutputPath.relative_to(ModelRoot)),
                    "Threshold": float(Args.Threshold),
                    "GroundTruthPixels": int(GroundTruth.sum()),
                    "PredictedPixels": int(Prediction.sum()),
                    **Metrics,
                }
            )

    IndexPath = TableDirectory / "PredictionFigureIndex.csv"
    AuditPath = AuditDirectory / "VisualizePredictionsAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    Result = pd.DataFrame(Rows)
    Result.to_csv(IndexPath, index=False)

    Audit = BuildAuditRecord(
        ScriptName="Step13VisualizePredictions.py",
        RunTag=Args.RunTag,
        Parameters=vars(Args),
        Inputs={
            "CaseSet": str(CasePath),
            "Checkpoint": str(CheckpointPath),
            "Features": str(FeaturePath),
            "Masks": str(MaskPath),
        },
        Outputs={
            "PredictionFigureIndex": str(IndexPath),
            "Figures": Result["FigurePath"].tolist() if not Result.empty else [],
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "ModelRunId": ModelRunId,
            "FiguresCreated": int(len(Result)),
            "CaseGroups": Args.CaseGroups,
        },
    )

    WriteJson(Audit, AuditPath)

    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=Args.RunTag,
        Step="Step13VisualizePredictions",
        Config=Args.FeatureConfig,
        Model=ModelRunId,
        OutputType="Table",
        RelativePath=str(IndexPath.relative_to(Paths.RunDirectory)),
        Created=IndexPath.exists(),
        Description=f"Índice de figuras de predicción {ModelRunId}.",
    )

    for _, Row in Result.iterrows():
        FigurePath = ModelRoot / Row["FigurePath"]
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step13VisualizePredictions",
            Config=Args.FeatureConfig,
            Model=ModelRunId,
            OutputType="Figure",
            RelativePath=str(FigurePath.relative_to(Paths.RunDirectory)),
            Created=FigurePath.exists(),
            Description=f"Figura de predicción {ModelRunId}.",
        )

    print("\n=== Prediction visualization completed ===")
    print("ModelRunId:", ModelRunId)
    print("Figures:", FigureDirectory)
    print("Index:", IndexPath)
    print(Result[["CaseGroup", "Order", "SampleId", "Dice", "IoU", "FigurePath"]].head(30).to_string(index=False))


if __name__ == "__main__":
    Main()
