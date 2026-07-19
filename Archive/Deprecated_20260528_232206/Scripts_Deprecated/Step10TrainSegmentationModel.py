#!/usr/bin/env python3
"""
Step10TrainSegmentationModel.py

Entrena una combinación específica:

RunTag + FeatureConfig + ModelName + RunName

Este script NO está asociado a una red única.
La arquitectura se selecciona mediante Source/Models/ModelFactory.py.

Ejemplo:
python Scripts/Step10TrainSegmentationModel.py \
  --RunTag Exp261944 \
  --FeatureConfig ConfigB \
  --ModelName EnhancedUNet \
  --RunName Smoke \
  --Epochs 1 \
  --BatchSize 2
"""

from __future__ import annotations

import argparse
import random
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
from Source.Losses import BceDiceLoss
from Source.Metrics import AggregateMetricRows, ComputeBinarySegmentationMetrics
from Source.Models.ModelFactory import CountTrainableParameters, CreateModel, ListAvailableModels
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


def BuildModelRunId(ModelName: str, RunName: str | None) -> str:
    """Construye identificador de ejecución del modelo."""
    if RunName is None or str(RunName).strip() == "":
        return ModelName

    CleanRunName = str(RunName).strip().replace(" ", "")
    return f"{ModelName}_{CleanRunName}"


def GetModelRunDirectories(RunDirectory: Path, FeatureConfig: str, ModelRunId: str) -> dict[str, Path]:
    """Crea y devuelve rutas estándar para un entrenamiento."""
    Root = RunDirectory / FeatureConfig / ModelRunId

    Directories = {
        "Root": Root,
        "Checkpoints": Root / "Checkpoints",
        "Metrics": Root / "Metrics",
        "Tables": Root / "Tables",
        "Figures": Root / "Figures",
        "Reports": Root / "Reports",
        "Audit": Root / "Audit",
    }

    for Directory in Directories.values():
        Directory.mkdir(parents=True, exist_ok=True)

    return Directories


def SetSeed(Seed: int) -> None:
    """Fija semillas básicas."""
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)


def ResolveDevice(DeviceArgument: str) -> torch.device:
    """Resuelve dispositivo de entrenamiento."""
    if DeviceArgument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Device = torch.device(DeviceArgument)

    if Device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero CUDA no está disponible.")

    return Device


def LoadFeatureConfig(ProjectRoot: Path, FeatureConfig: str) -> dict[str, Any]:
    """Carga ConfigA.yaml / ConfigB.yaml / ConfigC.yaml."""
    ConfigPath = ProjectRoot / "Configs" / f"{FeatureConfig}.yaml"

    if not ConfigPath.exists():
        raise FileNotFoundError(ConfigPath)

    Config = LoadYaml(ConfigPath)

    Required = ["FeatureConfig", "InputChannels", "Features"]
    Missing = [Key for Key in Required if Key not in Config]

    if Missing:
        raise KeyError(f"Faltan claves en {ConfigPath}: {Missing}")

    if Config["FeatureConfig"] != FeatureConfig:
        raise ValueError(
            f"{ConfigPath}: FeatureConfig={Config['FeatureConfig']} no coincide con {FeatureConfig}"
        )

    return Config


def BuildLoader(
    RunDirectory: Path,
    FeatureConfig: str,
    Split: str,
    ExpectedChannels: int,
    BatchSize: int,
    NumWorkers: int,
    Shuffle: bool,
) -> DataLoader:
    """Construye DataLoader para un split."""
    FeaturePath = RunDirectory / FeatureConfig / "Features" / f"{Split}Features.npy"
    MaskPath = RunDirectory / FeatureConfig / "Features" / f"{Split}Masks.npy"

    DatasetObject = FeatureTensorDataset(
        FeaturePath=FeaturePath,
        MaskPath=MaskPath,
        ExpectedChannels=ExpectedChannels,
        ExpectedHeight=200,
        ExpectedWidth=200,
    )

    Loader = DataLoader(
        DatasetObject,
        batch_size=BatchSize,
        shuffle=Shuffle,
        num_workers=NumWorkers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return Loader


def RunEpoch(
    Model: torch.nn.Module,
    Loader: DataLoader,
    LossFunction,
    Device: torch.device,
    Optimizer: torch.optim.Optimizer | None,
    Threshold: float,
    MaxBatches: int | None,
) -> dict[str, Any]:
    """Ejecuta una época de train o validation."""
    IsTraining = Optimizer is not None

    if IsTraining:
        Model.train()
    else:
        Model.eval()

    Rows = []

    for BatchIndex, Batch in enumerate(Loader):
        if MaxBatches is not None and BatchIndex >= MaxBatches:
            break

        Features = Batch["features"].to(Device, non_blocking=True)
        Masks = Batch["mask"].to(Device, non_blocking=True)

        if IsTraining:
            Optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(IsTraining):
            Logits = Model(Features)
            Loss = LossFunction(Logits, Masks)

            if IsTraining:
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=1.0)
                Optimizer.step()

        Metrics = ComputeBinarySegmentationMetrics(
            Logits=Logits.detach(),
            Targets=Masks.detach(),
            Threshold=Threshold,
        )

        Metrics["Loss"] = float(Loss.detach().cpu().item())
        Metrics["BatchIndex"] = int(BatchIndex)
        Rows.append(Metrics)

    return AggregateMetricRows(Rows)


def SaveCheckpoint(
    OutputPath: Path,
    Model: torch.nn.Module,
    Optimizer: torch.optim.Optimizer,
    Epoch: int,
    Metrics: dict[str, Any],
    Args,
    ModelRunId: str,
    ModelParameters: dict[str, Any],
) -> None:
    """Guarda checkpoint."""
    OutputPath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "Epoch": int(Epoch),
            "ModelRunId": ModelRunId,
            "ModelName": Args.ModelName,
            "RunName": Args.RunName,
            "FeatureConfig": Args.FeatureConfig,
            "ModelParameters": ModelParameters,
            "ModelStateDict": Model.state_dict(),
            "OptimizerStateDict": Optimizer.state_dict(),
            "Metrics": Metrics,
            "Args": vars(Args),
        },
        OutputPath,
    )


def Main() -> None:
    Parser = argparse.ArgumentParser(
        description="Entrena un modelo de segmentación seleccionado por ModelFactory."
    )
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--FeatureConfig", required=True, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--ModelName", required=True, choices=ListAvailableModels())
    Parser.add_argument("--RunName", default="Default")

    Parser.add_argument("--Epochs", type=int, default=30)
    Parser.add_argument("--BatchSize", type=int, default=4)
    Parser.add_argument("--NumWorkers", type=int, default=0)

    Parser.add_argument("--LearningRate", type=float, default=1e-4)
    Parser.add_argument("--WeightDecay", type=float, default=1e-5)

    Parser.add_argument("--BaseChannels", type=int, default=32)
    Parser.add_argument("--ReflectPadding", type=int, default=4)
    Parser.add_argument("--Dropout", type=float, default=0.0)
    Parser.add_argument("--UseSqueezeExcitation", action="store_true")
    Parser.add_argument("--TransformerHeads", type=int, default=8)
    Parser.add_argument("--TransformerLayers", type=int, default=2)
    Parser.add_argument("--TransformerMlpRatio", type=float, default=4.0)

    Parser.add_argument("--BceWeight", type=float, default=0.5)
    Parser.add_argument("--DiceWeight", type=float, default=0.5)
    Parser.add_argument("--Threshold", type=float, default=0.5)

    Parser.add_argument("--Device", default="auto")
    Parser.add_argument("--Seed", type=int, default=42)

    Parser.add_argument("--MaxTrainBatches", type=int, default=None)
    Parser.add_argument("--MaxValidationBatches", type=int, default=None)

    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    if Args.Epochs <= 0:
        raise ValueError("--Epochs debe ser positivo.")

    SetSeed(Args.Seed)

    Paths = CreateExperimentDirectories(Args.RunTag)

    ModelRunId = BuildModelRunId(Args.ModelName, Args.RunName)
    RunDirs = GetModelRunDirectories(
        RunDirectory=Paths.RunDirectory,
        FeatureConfig=Args.FeatureConfig,
        ModelRunId=ModelRunId,
    )

    Logger = CreateLogger(
        f"Step10TrainSegmentationModel_{Args.FeatureConfig}_{ModelRunId}",
        Paths.LogsDirectory / f"Step10TrainSegmentationModel_{Args.FeatureConfig}_{ModelRunId}.log",
    )

    Device = ResolveDevice(Args.Device)

    FeatureConfigYaml = LoadFeatureConfig(Paths.ProjectRoot, Args.FeatureConfig)
    InputChannels = int(FeatureConfigYaml["InputChannels"])

    ModelParameters = {
        "BaseChannels": int(Args.BaseChannels),
        "ReflectPadding": int(Args.ReflectPadding),
        "UseSqueezeExcitation": bool(Args.UseSqueezeExcitation),
        "Dropout": float(Args.Dropout),
    }

    if Args.ModelName == "TransformerUNet":
        ModelParameters.update(
            {
                "TransformerHeads": int(Args.TransformerHeads),
                "TransformerLayers": int(Args.TransformerLayers),
                "TransformerMlpRatio": float(Args.TransformerMlpRatio),
            }
        )

    Model = CreateModel(
        ModelName=Args.ModelName,
        InputChannels=InputChannels,
        OutputChannels=1,
        ModelParameters=ModelParameters,
    ).to(Device)

    ParameterCount = CountTrainableParameters(Model)

    LossFunction = BceDiceLoss(
        BceWeight=float(Args.BceWeight),
        DiceWeight=float(Args.DiceWeight),
    )

    Optimizer = torch.optim.AdamW(
        Model.parameters(),
        lr=float(Args.LearningRate),
        weight_decay=float(Args.WeightDecay),
    )

    TrainLoader = BuildLoader(
        RunDirectory=Paths.RunDirectory,
        FeatureConfig=Args.FeatureConfig,
        Split="Train",
        ExpectedChannels=InputChannels,
        BatchSize=Args.BatchSize,
        NumWorkers=Args.NumWorkers,
        Shuffle=True,
    )

    ValidationLoader = BuildLoader(
        RunDirectory=Paths.RunDirectory,
        FeatureConfig=Args.FeatureConfig,
        Split="Validation",
        ExpectedChannels=InputChannels,
        BatchSize=Args.BatchSize,
        NumWorkers=Args.NumWorkers,
        Shuffle=False,
    )

    Logger.info("RunTag: %s", Args.RunTag)
    Logger.info("FeatureConfig: %s", Args.FeatureConfig)
    Logger.info("ModelName: %s", Args.ModelName)
    Logger.info("RunName: %s", Args.RunName)
    Logger.info("ModelRunId: %s", ModelRunId)
    Logger.info("InputChannels: %d", InputChannels)
    Logger.info("Device: %s", Device)
    Logger.info("TrainableParameters: %d", ParameterCount)
    Logger.info("OutputRoot: %s", RunDirs["Root"])

    BestValidationDice = -1.0
    BestEpoch = -1
    HistoryRows = []

    BestCheckpointPath = RunDirs["Checkpoints"] / "BestModel.pt"
    LastCheckpointPath = RunDirs["Checkpoints"] / "LastModel.pt"
    HistoryPath = RunDirs["Metrics"] / "TrainingHistory.csv"
    BestEpochSummaryPath = RunDirs["Metrics"] / "BestEpochSummary.csv"
    ModelRunSummaryPath = RunDirs["Tables"] / "ModelRunSummary.csv"
    AuditPath = RunDirs["Audit"] / "TrainModelAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    StartTime = time.time()

    for Epoch in range(1, Args.Epochs + 1):
        Logger.info("Epoch %d/%d", Epoch, Args.Epochs)

        TrainMetrics = RunEpoch(
            Model=Model,
            Loader=TrainLoader,
            LossFunction=LossFunction,
            Device=Device,
            Optimizer=Optimizer,
            Threshold=Args.Threshold,
            MaxBatches=Args.MaxTrainBatches,
        )

        ValidationMetrics = RunEpoch(
            Model=Model,
            Loader=ValidationLoader,
            LossFunction=LossFunction,
            Device=Device,
            Optimizer=None,
            Threshold=Args.Threshold,
            MaxBatches=Args.MaxValidationBatches,
        )

        Row = {
            "Epoch": int(Epoch),
            "RunTag": Args.RunTag,
            "FeatureConfig": Args.FeatureConfig,
            "ModelName": Args.ModelName,
            "RunName": Args.RunName,
            "ModelRunId": ModelRunId,
            "TrainLoss": TrainMetrics["MeanLoss"],
            "TrainMeanDice": TrainMetrics["MeanDice"],
            "TrainMeanIoU": TrainMetrics["MeanIoU"],
            "TrainMeanPrecision": TrainMetrics["MeanPrecision"],
            "TrainMeanRecall": TrainMetrics["MeanRecall"],
            "TrainGlobalDice": TrainMetrics["GlobalDice"],
            "TrainGlobalIoU": TrainMetrics["GlobalIoU"],
            "ValidationLoss": ValidationMetrics["MeanLoss"],
            "ValidationMeanDice": ValidationMetrics["MeanDice"],
            "ValidationMeanIoU": ValidationMetrics["MeanIoU"],
            "ValidationMeanPrecision": ValidationMetrics["MeanPrecision"],
            "ValidationMeanRecall": ValidationMetrics["MeanRecall"],
            "ValidationGlobalDice": ValidationMetrics["GlobalDice"],
            "ValidationGlobalIoU": ValidationMetrics["GlobalIoU"],
            "TrainBatches": TrainMetrics["Batches"],
            "ValidationBatches": ValidationMetrics["Batches"],
            "LearningRate": Args.LearningRate,
        }

        HistoryRows.append(Row)
        pd.DataFrame(HistoryRows).to_csv(HistoryPath, index=False)

        Logger.info(
            "Epoch %d | TrainLoss %.5f | ValLoss %.5f | ValDice %.5f | ValIoU %.5f",
            Epoch,
            Row["TrainLoss"],
            Row["ValidationLoss"],
            Row["ValidationMeanDice"],
            Row["ValidationMeanIoU"],
        )

        SaveCheckpoint(
            OutputPath=LastCheckpointPath,
            Model=Model,
            Optimizer=Optimizer,
            Epoch=Epoch,
            Metrics=Row,
            Args=Args,
            ModelRunId=ModelRunId,
            ModelParameters=ModelParameters,
        )

        if Row["ValidationMeanDice"] > BestValidationDice:
            BestValidationDice = float(Row["ValidationMeanDice"])
            BestEpoch = int(Epoch)

            SaveCheckpoint(
                OutputPath=BestCheckpointPath,
                Model=Model,
                Optimizer=Optimizer,
                Epoch=Epoch,
                Metrics=Row,
                Args=Args,
                ModelRunId=ModelRunId,
                ModelParameters=ModelParameters,
            )

            Logger.info(
                "New best checkpoint saved | Epoch %d | ValidationMeanDice %.5f",
                BestEpoch,
                BestValidationDice,
            )

    ElapsedSeconds = time.time() - StartTime

    HistoryTable = pd.DataFrame(HistoryRows)

    if HistoryTable.empty:
        raise RuntimeError("TrainingHistory está vacío.")

    BestRow = HistoryTable.sort_values(
        ["ValidationMeanDice", "ValidationMeanIoU"],
        ascending=[False, False],
    ).iloc[0].to_dict()

    BestEpochSummary = {
        "RunTag": Args.RunTag,
        "FeatureConfig": Args.FeatureConfig,
        "ModelName": Args.ModelName,
        "RunName": Args.RunName,
        "ModelRunId": ModelRunId,
        "BestEpoch": int(BestRow["Epoch"]),
        "BestValidationLoss": float(BestRow["ValidationLoss"]),
        "BestValidationMeanDice": float(BestRow["ValidationMeanDice"]),
        "BestValidationMeanIoU": float(BestRow["ValidationMeanIoU"]),
        "BestValidationGlobalDice": float(BestRow["ValidationGlobalDice"]),
        "BestValidationGlobalIoU": float(BestRow["ValidationGlobalIoU"]),
        "BestCheckpoint": str(BestCheckpointPath),
        "LastCheckpoint": str(LastCheckpointPath),
        "EpochsCompleted": int(Args.Epochs),
        "ElapsedSeconds": float(ElapsedSeconds),
    }

    pd.DataFrame([BestEpochSummary]).to_csv(BestEpochSummaryPath, index=False)

    ModelRunSummary = {
        "RunTag": Args.RunTag,
        "FeatureConfig": Args.FeatureConfig,
        "ModelName": Args.ModelName,
        "RunName": Args.RunName,
        "ModelRunId": ModelRunId,
        "InputChannels": int(InputChannels),
        "Features": ",".join(FeatureConfigYaml["Features"]),
        "Epochs": int(Args.Epochs),
        "BatchSize": int(Args.BatchSize),
        "BaseChannels": int(Args.BaseChannels),
        "LearningRate": float(Args.LearningRate),
        "WeightDecay": float(Args.WeightDecay),
        "Dropout": float(Args.Dropout),
        "UseSqueezeExcitation": bool(Args.UseSqueezeExcitation),
        "TransformerHeads": int(getattr(Args, "TransformerHeads", 0)),
        "TransformerLayers": int(getattr(Args, "TransformerLayers", 0)),
        "TransformerMlpRatio": float(getattr(Args, "TransformerMlpRatio", 0.0)),
        "BceWeight": float(Args.BceWeight),
        "DiceWeight": float(Args.DiceWeight),
        "Threshold": float(Args.Threshold),
        "TrainableParameters": int(ParameterCount),
        "Device": str(Device),
        "OutputRoot": str(RunDirs["Root"]),
        "BestEpoch": int(BestEpochSummary["BestEpoch"]),
        "BestValidationMeanDice": float(BestEpochSummary["BestValidationMeanDice"]),
        "BestValidationMeanIoU": float(BestEpochSummary["BestValidationMeanIoU"]),
    }

    pd.DataFrame([ModelRunSummary]).to_csv(ModelRunSummaryPath, index=False)

    Audit = BuildAuditRecord(
        ScriptName="Step10TrainSegmentationModel.py",
        RunTag=Args.RunTag,
        Parameters={
            **vars(Args),
            "ModelRunId": ModelRunId,
            "InputChannels": InputChannels,
            "Features": FeatureConfigYaml["Features"],
            "ModelParameters": ModelParameters,
            "TrainableParameters": ParameterCount,
            "DeviceResolved": str(Device),
        },
        Inputs={
            "TrainFeatures": str(
                Paths.RunDirectory / Args.FeatureConfig / "Features" / "TrainFeatures.npy"
            ),
            "TrainMasks": str(
                Paths.RunDirectory / Args.FeatureConfig / "Features" / "TrainMasks.npy"
            ),
            "ValidationFeatures": str(
                Paths.RunDirectory / Args.FeatureConfig / "Features" / "ValidationFeatures.npy"
            ),
            "ValidationMasks": str(
                Paths.RunDirectory / Args.FeatureConfig / "Features" / "ValidationMasks.npy"
            ),
        },
        Outputs={
            "BestCheckpoint": str(BestCheckpointPath),
            "LastCheckpoint": str(LastCheckpointPath),
            "TrainingHistory": str(HistoryPath),
            "BestEpochSummary": str(BestEpochSummaryPath),
            "ModelRunSummary": str(ModelRunSummaryPath),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "BestValidationMeanDice": float(BestValidationDice),
            "BestEpoch": int(BestEpoch),
            "EpochsCompleted": int(Args.Epochs),
            "ElapsedSeconds": float(ElapsedSeconds),
            "InputShape": "B,C,200,200",
            "ReflectPadding": int(Args.ReflectPadding),
            "InternalShape": "B,C,208,208",
            "OutputShape": "B,1,200,200",
        },
    )

    WriteJson(Audit, AuditPath)

    for OutputType, OutputPath, Description in [
        ("Checkpoint", BestCheckpointPath, f"Mejor checkpoint {ModelRunId}."),
        ("Checkpoint", LastCheckpointPath, f"Último checkpoint {ModelRunId}."),
        ("Table", HistoryPath, f"Historial de entrenamiento {ModelRunId}."),
        ("Table", BestEpochSummaryPath, f"Resumen de mejor época {ModelRunId}."),
        ("Table", ModelRunSummaryPath, f"Resumen técnico de ejecución {ModelRunId}."),
        ("Audit", AuditPath, f"Auditoría de entrenamiento {ModelRunId}."),
    ]:
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step10TrainSegmentationModel",
            Config=Args.FeatureConfig,
            Model=ModelRunId,
            OutputType=OutputType,
            RelativePath=str(OutputPath.relative_to(Paths.RunDirectory)),
            Created=OutputPath.exists(),
            Description=Description,
        )

    print("\n=== Training completed ===")
    print("RunTag:", Args.RunTag)
    print("FeatureConfig:", Args.FeatureConfig)
    print("ModelName:", Args.ModelName)
    print("RunName:", Args.RunName)
    print("ModelRunId:", ModelRunId)
    print("BestEpoch:", BestEpoch)
    print("BestValidationMeanDice:", BestValidationDice)
    print("OutputRoot:", RunDirs["Root"])
    print("History:", HistoryPath)
    print("BestEpochSummary:", BestEpochSummaryPath)
    print("ModelRunSummary:", ModelRunSummaryPath)
    print("BestCheckpoint:", BestCheckpointPath)


if __name__ == "__main__":
    Main()
