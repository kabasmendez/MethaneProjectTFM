#!/usr/bin/env python3
"""
Step11EvaluateSegmentationModel.py

Evaluación formal de un modelo de segmentación entrenado.

Evalúa una combinación específica:

RunTag + FeatureConfig + ModelName + RunName

Entradas:
- <FeatureConfig>/Features/TestFeatures.npy
- <FeatureConfig>/Features/TestMasks.npy
- <FeatureConfig>/<ModelRunId>/Checkpoints/BestModel.pt o LastModel.pt
- Tables/SplitTestFeatureReady.csv

Salidas:
- <FeatureConfig>/<ModelRunId>/Metrics/TestMetricsSummary.csv
- <FeatureConfig>/<ModelRunId>/Metrics/TestMetricsBySample.csv
- <FeatureConfig>/<ModelRunId>/Audit/EvaluateModelAudit.json

Notas:
- Evalúa todo el split Test FeatureReady.
- Guarda métricas por muestra para análisis posterior, visualización y comparación.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml
from Source.FeatureTensorDataset import FeatureTensorDataset
from Source.LoggingUtils import CreateLogger
from Source.Models.ModelFactory import CreateModel
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


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
        raise RuntimeError("Se solicitó CUDA, pero CUDA no está disponible.")

    return Device


def LoadFeatureConfig(ProjectRoot: Path, FeatureConfig: str) -> dict[str, Any]:
    """Carga archivo ConfigA.yaml / ConfigB.yaml / ConfigC.yaml."""
    ConfigPath = ProjectRoot / "Configs" / f"{FeatureConfig}.yaml"

    if not ConfigPath.exists():
        raise FileNotFoundError(ConfigPath)

    Config = LoadYaml(ConfigPath)

    Required = ["FeatureConfig", "InputChannels", "Features"]
    Missing = [Key for Key in Required if Key not in Config]

    if Missing:
        raise KeyError(f"Faltan claves en {ConfigPath}: {Missing}")

    return Config


def LoadSplitSampleIds(RunDirectory: Path, Split: str) -> list[str]:
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


@torch.no_grad()
def ComputePerSampleMetrics(
    Logits: torch.Tensor,
    Masks: torch.Tensor,
    Threshold: float,
    Epsilon: float = 1e-7,
) -> list[dict[str, Any]]:
    """Calcula métricas por muestra dentro de un batch."""
    Probabilities = torch.sigmoid(Logits)
    Predictions = Probabilities >= Threshold
    Targets = Masks >= 0.5

    B = int(Logits.shape[0])
    Rows = []

    for Index in range(B):
        Pred = Predictions[Index].reshape(-1)
        Target = Targets[Index].reshape(-1)

        TP = int((Pred & Target).sum().item())
        FP = int((Pred & ~Target).sum().item())
        FN = int((~Pred & Target).sum().item())
        TN = int((~Pred & ~Target).sum().item())

        Dice = (2.0 * TP + Epsilon) / (2.0 * TP + FP + FN + Epsilon)
        IoU = (TP + Epsilon) / (TP + FP + FN + Epsilon)
        Precision = (TP + Epsilon) / (TP + FP + Epsilon)
        Recall = (TP + Epsilon) / (TP + FN + Epsilon)

        GroundTruthPixels = int(Target.sum().item())
        PredictedPixels = int(Pred.sum().item())

        Rows.append(
            {
                "Dice": float(Dice),
                "IoU": float(IoU),
                "Precision": float(Precision),
                "Recall": float(Recall),
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "GroundTruthPixels": GroundTruthPixels,
                "PredictedPixels": PredictedPixels,
                "FalsePositivePixels": FP,
                "FalseNegativePixels": FN,
                "ProbabilityMean": float(Probabilities[Index].mean().item()),
                "ProbabilityMax": float(Probabilities[Index].max().item()),
            }
        )

    return Rows


def BuildSummary(BySample: pd.DataFrame) -> dict[str, Any]:
    """Construye resumen global del test."""
    TotalTP = int(BySample["TP"].sum())
    TotalFP = int(BySample["FP"].sum())
    TotalFN = int(BySample["FN"].sum())
    TotalTN = int(BySample["TN"].sum())

    Epsilon = 1e-7

    GlobalDice = (2.0 * TotalTP + Epsilon) / (
        2.0 * TotalTP + TotalFP + TotalFN + Epsilon
    )
    GlobalIoU = (TotalTP + Epsilon) / (
        TotalTP + TotalFP + TotalFN + Epsilon
    )
    GlobalPrecision = (TotalTP + Epsilon) / (TotalTP + TotalFP + Epsilon)
    GlobalRecall = (TotalTP + Epsilon) / (TotalTP + TotalFN + Epsilon)

    return {
        "Samples": int(len(BySample)),
        "MeanDice": float(BySample["Dice"].mean()),
        "MedianDice": float(BySample["Dice"].median()),
        "MeanIoU": float(BySample["IoU"].mean()),
        "MedianIoU": float(BySample["IoU"].median()),
        "MeanPrecision": float(BySample["Precision"].mean()),
        "MeanRecall": float(BySample["Recall"].mean()),
        "GlobalDice": float(GlobalDice),
        "GlobalIoU": float(GlobalIoU),
        "GlobalPrecision": float(GlobalPrecision),
        "GlobalRecall": float(GlobalRecall),
        "TP": TotalTP,
        "FP": TotalFP,
        "FN": TotalFN,
        "TN": TotalTN,
        "MeanGroundTruthPixels": float(BySample["GroundTruthPixels"].mean()),
        "MeanPredictedPixels": float(BySample["PredictedPixels"].mean()),
        "TotalGroundTruthPixels": int(BySample["GroundTruthPixels"].sum()),
        "TotalPredictedPixels": int(BySample["PredictedPixels"].sum()),
    }


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Evalúa formalmente un modelo en Test.")
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--FeatureConfig", required=True, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--ModelName", required=True)
    Parser.add_argument("--RunName", required=True)
    Parser.add_argument("--Split", default="Test", choices=["Train", "Validation", "Test"])
    Parser.add_argument("--Checkpoint", default="BestModel.pt", choices=["BestModel.pt", "LastModel.pt"])
    Parser.add_argument("--BatchSize", type=int, default=8)
    Parser.add_argument("--NumWorkers", type=int, default=0)
    Parser.add_argument("--Threshold", type=float, default=0.5)
    Parser.add_argument("--Device", default="auto")

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    Device = ResolveDevice(Args.Device)

    ModelRunId = BuildModelRunId(Args.ModelName, Args.RunName)
    ModelRoot = Paths.RunDirectory / Args.FeatureConfig / ModelRunId

    MetricsDirectory = ModelRoot / "Metrics"
    AuditDirectory = ModelRoot / "Audit"
    TablesDirectory = ModelRoot / "Tables"

    MetricsDirectory.mkdir(parents=True, exist_ok=True)
    AuditDirectory.mkdir(parents=True, exist_ok=True)
    TablesDirectory.mkdir(parents=True, exist_ok=True)

    Logger = CreateLogger(
        f"Step11Evaluate_{Args.FeatureConfig}_{ModelRunId}",
        Paths.LogsDirectory / f"Step11Evaluate_{Args.FeatureConfig}_{ModelRunId}.log",
    )

    FeatureConfigYaml = LoadFeatureConfig(Paths.ProjectRoot, Args.FeatureConfig)
    InputChannels = int(FeatureConfigYaml["InputChannels"])

    FeaturePath = Paths.RunDirectory / Args.FeatureConfig / "Features" / f"{Args.Split}Features.npy"
    MaskPath = Paths.RunDirectory / Args.FeatureConfig / "Features" / f"{Args.Split}Masks.npy"
    CheckpointPath = ModelRoot / "Checkpoints" / Args.Checkpoint

    if not CheckpointPath.exists():
        raise FileNotFoundError(f"No existe checkpoint: {CheckpointPath}")

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
        num_workers=Args.NumWorkers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    SampleIds = LoadSplitSampleIds(Paths.RunDirectory, Args.Split)

    if len(SampleIds) != len(DatasetObject):
        raise ValueError(
            f"Cantidad de SampleIds no coincide con dataset: "
            f"{len(SampleIds)} vs {len(DatasetObject)}"
        )

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

    Logger.info("Evaluating %s on %s", ModelRunId, Args.Split)
    Logger.info("Checkpoint: %s", CheckpointPath)
    Logger.info("Device: %s", Device)

    Rows = []
    StartTime = time.time()

    with torch.no_grad():
        for BatchIndex, Batch in enumerate(Loader):
            Features = Batch["features"].to(Device, non_blocking=True)
            Masks = Batch["mask"].to(Device, non_blocking=True)

            Logits = Model(Features)

            BatchRows = ComputePerSampleMetrics(
                Logits=Logits,
                Masks=Masks,
                Threshold=Args.Threshold,
            )

            for LocalIndex, MetricRow in enumerate(BatchRows):
                DatasetIndex = int(Batch["index"][LocalIndex].item())
                SampleId = SampleIds[DatasetIndex]

                Rows.append(
                    {
                        "RunTag": Args.RunTag,
                        "FeatureConfig": Args.FeatureConfig,
                        "ModelName": Args.ModelName,
                        "RunName": Args.RunName,
                        "ModelRunId": ModelRunId,
                        "Split": Args.Split,
                        "DatasetIndex": DatasetIndex,
                        "SampleId": SampleId,
                        "Checkpoint": Args.Checkpoint,
                        "Threshold": float(Args.Threshold),
                        **MetricRow,
                    }
                )

            if (BatchIndex + 1) % 20 == 0:
                Logger.info("Processed batches: %d/%d", BatchIndex + 1, len(Loader))

    ElapsedSeconds = time.time() - StartTime

    BySample = pd.DataFrame(Rows)

    if BySample.empty:
        raise RuntimeError("No se generaron métricas por muestra.")

    Summary = BuildSummary(BySample)
    Summary.update(
        {
            "RunTag": Args.RunTag,
            "FeatureConfig": Args.FeatureConfig,
            "ModelName": Args.ModelName,
            "RunName": Args.RunName,
            "ModelRunId": ModelRunId,
            "Split": Args.Split,
            "Checkpoint": Args.Checkpoint,
            "Threshold": float(Args.Threshold),
            "ElapsedSeconds": float(ElapsedSeconds),
            "CheckpointEpoch": int(Checkpoint.get("Epoch", -1)),
        }
    )

    BySamplePath = MetricsDirectory / f"{Args.Split}MetricsBySample.csv"
    SummaryPath = MetricsDirectory / f"{Args.Split}MetricsSummary.csv"
    AuditPath = AuditDirectory / "EvaluateModelAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    BySample.to_csv(BySamplePath, index=False)
    pd.DataFrame([Summary]).to_csv(SummaryPath, index=False)

    Audit = BuildAuditRecord(
        ScriptName="Step11EvaluateSegmentationModel.py",
        RunTag=Args.RunTag,
        Parameters=vars(Args),
        Inputs={
            "Checkpoint": str(CheckpointPath),
            "Features": str(FeaturePath),
            "Masks": str(MaskPath),
            "SplitSampleIds": str(Paths.RunDirectory / "Tables" / f"Split{Args.Split}FeatureReady.csv"),
        },
        Outputs={
            "MetricsBySample": str(BySamplePath),
            "MetricsSummary": str(SummaryPath),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "ModelRunId": ModelRunId,
            "SamplesEvaluated": int(len(BySample)),
            "MeanDice": float(Summary["MeanDice"]),
            "MeanIoU": float(Summary["MeanIoU"]),
            "GlobalDice": float(Summary["GlobalDice"]),
            "GlobalIoU": float(Summary["GlobalIoU"]),
        },
    )

    WriteJson(Audit, AuditPath)

    for OutputType, OutputPath, Description in [
        ("Table", BySamplePath, f"Métricas por muestra {Args.Split} para {ModelRunId}."),
        ("Table", SummaryPath, f"Resumen de métricas {Args.Split} para {ModelRunId}."),
        ("Audit", AuditPath, f"Auditoría de evaluación para {ModelRunId}."),
    ]:
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step11EvaluateSegmentationModel",
            Config=Args.FeatureConfig,
            Model=ModelRunId,
            OutputType=OutputType,
            RelativePath=str(OutputPath.relative_to(Paths.RunDirectory)),
            Created=OutputPath.exists(),
            Description=Description,
        )

    print("\n=== Evaluation completed ===")
    print("RunTag:", Args.RunTag)
    print("FeatureConfig:", Args.FeatureConfig)
    print("ModelRunId:", ModelRunId)
    print("Split:", Args.Split)
    print("Checkpoint:", CheckpointPath)
    print("MetricsBySample:", BySamplePath)
    print("MetricsSummary:", SummaryPath)
    print(pd.DataFrame([Summary])[[
        "Samples",
        "MeanDice",
        "MeanIoU",
        "GlobalDice",
        "GlobalIoU",
        "MeanPrecision",
        "MeanRecall",
    ]].to_string(index=False))


if __name__ == "__main__":
    Main()
