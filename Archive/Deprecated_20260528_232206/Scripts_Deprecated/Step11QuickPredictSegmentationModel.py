#!/usr/bin/env python3
"""
Step11QuickPredictSegmentationModel.py

Genera predicciones rápidas para verificar que un modelo entrenado produce salida.

Este script:
- carga BestModel.pt o LastModel.pt;
- lee tensores TestFeatures.npy y TestMasks.npy;
- predice probabilidades;
- guarda figuras PNG;
- guarda métricas por muestra en CSV.

Uso:
python Scripts/Step11QuickPredictSegmentationModel.py \
  --RunTag Exp261944 \
  --FeatureConfig ConfigB \
  --ModelName EnhancedUNet \
  --RunName Ep5Preview \
  --MaxSamples 6
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
from Source.ConfigUtils import LoadYaml
from Source.FeatureTensorDataset import FeatureTensorDataset
from Source.LoggingUtils import CreateLogger
from Source.Metrics import ComputeBinarySegmentationMetrics
from Source.Models.ModelFactory import CreateModel
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments
from Source.VisualizationStyle import ApplyMatplotlibStyle, LoadVisualizationConfig


def BuildModelRunId(ModelName: str, RunName: str | None) -> str:
    """Construye identificador de ejecución del modelo."""
    if RunName is None or str(RunName).strip() == "":
        return ModelName
    CleanRunName = str(RunName).strip().replace(" ", "")
    return f"{ModelName}_{CleanRunName}"


def ResolveDevice(DeviceArgument: str) -> torch.device:
    """Resuelve dispositivo."""
    if DeviceArgument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Device = torch.device(DeviceArgument)

    if Device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero no está disponible.")

    return Device


def LoadFeatureConfig(ProjectRoot: Path, FeatureConfig: str) -> dict[str, Any]:
    """Carga configuración de features."""
    ConfigPath = ProjectRoot / "Configs" / f"{FeatureConfig}.yaml"

    if not ConfigPath.exists():
        raise FileNotFoundError(ConfigPath)

    Config = LoadYaml(ConfigPath)

    Required = ["FeatureConfig", "InputChannels", "Features"]
    Missing = [Key for Key in Required if Key not in Config]

    if Missing:
        raise KeyError(f"Faltan claves en {ConfigPath}: {Missing}")

    return Config


def LoadSampleIds(RunDirectory: Path, Split: str) -> list[str]:
    """Carga SampleId del split FeatureReady."""
    SplitPath = RunDirectory / "Tables" / f"Split{Split}FeatureReady.csv"

    if not SplitPath.exists():
        raise FileNotFoundError(
            f"No existe {SplitPath}. Ejecuta Step06ApplyFeatureReadinessFilter.py."
        )

    Table = pd.read_csv(SplitPath)

    if "SampleId" not in Table.columns:
        raise KeyError(f"{SplitPath} debe contener SampleId.")

    return Table["SampleId"].astype(str).tolist()


def ComputeErrorMap(GroundTruth: np.ndarray, Prediction: np.ndarray) -> np.ndarray:
    """
    Error map:
    0 = TN
    1 = TP
    2 = FP
    3 = FN
    """
    GroundTruth = GroundTruth.astype(bool)
    Prediction = Prediction.astype(bool)

    ErrorMap = np.zeros(GroundTruth.shape, dtype=np.uint8)
    ErrorMap[GroundTruth & Prediction] = 1
    ErrorMap[~GroundTruth & Prediction] = 2
    ErrorMap[GroundTruth & ~Prediction] = 3

    return ErrorMap


def SavePredictionFigure(
    OutputPath: Path,
    SampleId: str,
    Probability: np.ndarray,
    Prediction: np.ndarray,
    GroundTruth: np.ndarray,
    Metrics: dict[str, Any],
    VisualConfig: dict[str, Any],
) -> None:
    """Guarda figura de predicción rápida."""
    OutputPath.parent.mkdir(parents=True, exist_ok=True)

    ErrorMap = ComputeErrorMap(GroundTruth, Prediction)

    ErrorCmap = ListedColormap([
        "#000000",  # TN
        "#2ca02c",  # TP
        "#ff7f0e",  # FP
        "#d62728",  # FN
    ])

    Figure, Axes = plt.subplots(1, 4, figsize=(16, 4.5))

    Axes[0].imshow(GroundTruth, cmap="gray", vmin=0, vmax=1)
    Axes[0].set_title("Ground truth")

    Im1 = Axes[1].imshow(Probability, cmap="viridis", vmin=0, vmax=1)
    Axes[1].set_title("Predicted probability")
    Figure.colorbar(Im1, ax=Axes[1], fraction=0.046, pad=0.04)

    Axes[2].imshow(Prediction, cmap="gray", vmin=0, vmax=1)
    Axes[2].set_title("Prediction thresholded")

    Axes[3].imshow(ErrorMap, cmap=ErrorCmap, vmin=0, vmax=3)
    Axes[3].set_title("Error map: TP/FP/FN")

    for Axis in Axes:
        Axis.set_xticks([])
        Axis.set_yticks([])

    Title = (
        f"SampleId: {SampleId}\n"
        f"Dice={Metrics['MeanDice']:.4f} | IoU={Metrics['MeanIoU']:.4f} | "
        f"Precision={Metrics['MeanPrecision']:.4f} | Recall={Metrics['MeanRecall']:.4f}"
    )

    Figure.suptitle(Title)
    Figure.tight_layout(rect=[0, 0.02, 1, 0.90])

    Figure.savefig(
        OutputPath,
        dpi=VisualConfig["Visualization"].get("Dpi", 300),
        bbox_inches="tight",
        facecolor=VisualConfig["Visualization"].get("FigureFaceColor", "white"),
    )

    plt.close(Figure)


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Predicción rápida para un modelo entrenado.")
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--FeatureConfig", required=True, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--ModelName", required=True)
    Parser.add_argument("--RunName", required=True)
    Parser.add_argument("--Split", default="Test", choices=["Train", "Validation", "Test"])
    Parser.add_argument("--Checkpoint", default="BestModel.pt", choices=["BestModel.pt", "LastModel.pt"])
    Parser.add_argument("--MaxSamples", type=int, default=6)
    Parser.add_argument("--BatchSize", type=int, default=1)
    Parser.add_argument("--Threshold", type=float, default=0.5)
    Parser.add_argument("--Device", default="auto")
    Parser.add_argument(
        "--VisualizationConfig",
        default="Configs/VisualizationConfig.yaml",
    )

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    if Args.MaxSamples <= 0:
        raise ValueError("--MaxSamples debe ser positivo.")

    Paths = CreateExperimentDirectories(Args.RunTag)
    ModelRunId = BuildModelRunId(Args.ModelName, Args.RunName)

    ModelRoot = Paths.RunDirectory / Args.FeatureConfig / ModelRunId
    FigureDirectory = ModelRoot / "Figures" / "QuickPredictions"
    TableDirectory = ModelRoot / "Tables"
    AuditDirectory = ModelRoot / "Audit"

    FigureDirectory.mkdir(parents=True, exist_ok=True)
    TableDirectory.mkdir(parents=True, exist_ok=True)
    AuditDirectory.mkdir(parents=True, exist_ok=True)

    Logger = CreateLogger(
        f"Step11QuickPredict_{Args.FeatureConfig}_{ModelRunId}",
        Paths.LogsDirectory / f"Step11QuickPredict_{Args.FeatureConfig}_{ModelRunId}.log",
    )

    Device = ResolveDevice(Args.Device)

    VisualConfig = LoadVisualizationConfig(Paths.ProjectRoot / Args.VisualizationConfig)
    ApplyMatplotlibStyle(VisualConfig)

    FeatureConfigYaml = LoadFeatureConfig(Paths.ProjectRoot, Args.FeatureConfig)
    InputChannels = int(FeatureConfigYaml["InputChannels"])

    FeaturePath = Paths.RunDirectory / Args.FeatureConfig / "Features" / f"{Args.Split}Features.npy"
    MaskPath = Paths.RunDirectory / Args.FeatureConfig / "Features" / f"{Args.Split}Masks.npy"

    DatasetObject = FeatureTensorDataset(
        FeaturePath=FeaturePath,
        MaskPath=MaskPath,
        ExpectedChannels=InputChannels,
        ExpectedHeight=200,
        ExpectedWidth=200,
    )

    Loader = DataLoader(
        DatasetObject,
        batch_size=Args.BatchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    SampleIds = LoadSampleIds(Paths.RunDirectory, Args.Split)

    CheckpointPath = ModelRoot / "Checkpoints" / Args.Checkpoint

    if not CheckpointPath.exists():
        raise FileNotFoundError(f"No existe checkpoint: {CheckpointPath}")

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

    Rows = []
    SavedFigures = []
    Processed = 0

    Logger.info("Starting quick predictions.")
    Logger.info("Checkpoint: %s", CheckpointPath)

    with torch.no_grad():
        for Batch in Loader:
            Features = Batch["features"].to(Device)
            Masks = Batch["mask"].to(Device)

            Logits = Model(Features)
            Probabilities = torch.sigmoid(Logits)
            Predictions = Probabilities >= Args.Threshold

            BatchSize = Features.shape[0]

            for LocalIndex in range(BatchSize):
                DatasetIndex = int(Batch["index"][LocalIndex].item())
                SampleId = SampleIds[DatasetIndex]

                OneLogit = Logits[LocalIndex:LocalIndex + 1]
                OneMask = Masks[LocalIndex:LocalIndex + 1]

                Metrics = ComputeBinarySegmentationMetrics(
                    Logits=OneLogit,
                    Targets=OneMask,
                    Threshold=Args.Threshold,
                )

                Probability = Probabilities[LocalIndex, 0].detach().cpu().numpy()
                Prediction = Predictions[LocalIndex, 0].detach().cpu().numpy().astype(np.uint8)
                GroundTruth = Masks[LocalIndex, 0].detach().cpu().numpy().astype(np.uint8)

                OutputPath = FigureDirectory / f"QuickPrediction_{Processed:03d}_{SampleId}.png"

                SavePredictionFigure(
                    OutputPath=OutputPath,
                    SampleId=SampleId,
                    Probability=Probability,
                    Prediction=Prediction,
                    GroundTruth=GroundTruth,
                    Metrics=Metrics,
                    VisualConfig=VisualConfig,
                )

                SavedFigures.append(OutputPath)

                Rows.append({
                    "Split": Args.Split,
                    "DatasetIndex": DatasetIndex,
                    "SampleId": SampleId,
                    "FigurePath": str(OutputPath.relative_to(ModelRoot)),
                    "Threshold": float(Args.Threshold),
                    "Dice": Metrics["MeanDice"],
                    "IoU": Metrics["MeanIoU"],
                    "Precision": Metrics["MeanPrecision"],
                    "Recall": Metrics["MeanRecall"],
                    "TP": Metrics["TP"],
                    "FP": Metrics["FP"],
                    "FN": Metrics["FN"],
                    "TN": Metrics["TN"],
                    "GroundTruthPixels": int(GroundTruth.sum()),
                    "PredictedPixels": int(Prediction.sum()),
                })

                Processed += 1

                if Processed >= Args.MaxSamples:
                    break

            if Processed >= Args.MaxSamples:
                break

    MetricsPath = TableDirectory / "QuickPredictionMetrics.csv"
    AuditPath = AuditDirectory / "QuickPredictionAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    pd.DataFrame(Rows).to_csv(MetricsPath, index=False)

    Audit = BuildAuditRecord(
        ScriptName="Step11QuickPredictSegmentationModel.py",
        RunTag=Args.RunTag,
        Parameters=vars(Args),
        Inputs={
            "Checkpoint": str(CheckpointPath),
            "Features": str(FeaturePath),
            "Masks": str(MaskPath),
        },
        Outputs={
            "QuickPredictionMetrics": str(MetricsPath),
            "Figures": [str(PathItem) for PathItem in SavedFigures],
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "ModelRunId": ModelRunId,
            "ProcessedSamples": int(Processed),
            "InputChannels": int(InputChannels),
            "CheckpointEpoch": int(Checkpoint.get("Epoch", -1)),
        },
    )

    WriteJson(Audit, AuditPath)

    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=Args.RunTag,
        Step="Step11QuickPredictSegmentationModel",
        Config=Args.FeatureConfig,
        Model=ModelRunId,
        OutputType="Table",
        RelativePath=str(MetricsPath.relative_to(Paths.RunDirectory)),
        Created=MetricsPath.exists(),
        Description=f"Métricas rápidas de predicción {ModelRunId}.",
    )

    for FigurePath in SavedFigures:
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step11QuickPredictSegmentationModel",
            Config=Args.FeatureConfig,
            Model=ModelRunId,
            OutputType="Figure",
            RelativePath=str(FigurePath.relative_to(Paths.RunDirectory)),
            Created=FigurePath.exists(),
            Description=f"Figura rápida de predicción {ModelRunId}.",
        )

    print("\n=== Quick prediction completed ===")
    print("RunTag:", Args.RunTag)
    print("FeatureConfig:", Args.FeatureConfig)
    print("ModelRunId:", ModelRunId)
    print("Checkpoint:", CheckpointPath)
    print("Metrics:", MetricsPath)
    print("Figures:", FigureDirectory)
    print(pd.DataFrame(Rows)[["SampleId", "Dice", "IoU", "Precision", "Recall", "GroundTruthPixels", "PredictedPixels"]].to_string(index=False))


if __name__ == "__main__":
    Main()
