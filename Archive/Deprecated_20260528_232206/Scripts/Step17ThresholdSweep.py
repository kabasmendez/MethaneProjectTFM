#!/usr/bin/env python3
"""
Step17ThresholdSweep.py

Barrido de umbrales para un modelo ya entrenado.

Evalúa umbrales entre MinThreshold y MaxThreshold usando el split elegido.
Guarda:
- Metrics/<Split>ThresholdSweep.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.ConfigUtils import LoadYaml
from Source.FeatureTensorDataset import FeatureTensorDataset
from Source.Models.ModelFactory import CreateModel
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


def BuildModelRunId(ModelName: str, RunName: str | None) -> str:
    if RunName is None or str(RunName).strip() == "":
        return ModelName
    return f"{ModelName}_{str(RunName).strip().replace(' ', '')}"


def ResolveDevice(DeviceArgument: str) -> torch.device:
    if DeviceArgument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DeviceArgument)


def LoadFeatureConfig(ProjectRoot: Path, FeatureConfig: str) -> dict[str, Any]:
    ConfigPath = ProjectRoot / "Configs" / f"{FeatureConfig}.yaml"
    if not ConfigPath.exists():
        raise FileNotFoundError(ConfigPath)
    return LoadYaml(ConfigPath)


def ComputeMetricsFromCounts(TP: int, FP: int, FN: int, TN: int) -> dict[str, float]:
    Eps = 1e-7
    Dice = (2 * TP + Eps) / (2 * TP + FP + FN + Eps)
    IoU = (TP + Eps) / (TP + FP + FN + Eps)
    Precision = (TP + Eps) / (TP + FP + Eps)
    Recall = (TP + Eps) / (TP + FN + Eps)

    return {
        "GlobalDice": float(Dice),
        "GlobalIoU": float(IoU),
        "GlobalPrecision": float(Precision),
        "GlobalRecall": float(Recall),
    }


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Threshold sweep para modelo de segmentación.")
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--FeatureConfig", required=True, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--ModelName", required=True)
    Parser.add_argument("--RunName", required=True)
    Parser.add_argument("--Split", default="Validation", choices=["Train", "Validation", "Test"])
    Parser.add_argument("--Checkpoint", default="BestModel.pt", choices=["BestModel.pt", "LastModel.pt"])
    Parser.add_argument("--BatchSize", type=int, default=8)
    Parser.add_argument("--Device", default="auto")
    Parser.add_argument("--MinThreshold", type=float, default=0.05)
    Parser.add_argument("--MaxThreshold", type=float, default=0.95)
    Parser.add_argument("--NumThresholds", type=int, default=19)

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    Device = ResolveDevice(Args.Device)

    ModelRunId = BuildModelRunId(Args.ModelName, Args.RunName)
    ModelRoot = Paths.RunDirectory / Args.FeatureConfig / ModelRunId

    FeatureConfigYaml = LoadFeatureConfig(Paths.ProjectRoot, Args.FeatureConfig)
    InputChannels = int(FeatureConfigYaml["InputChannels"])

    FeaturePath = Paths.RunDirectory / Args.FeatureConfig / "Features" / f"{Args.Split}Features.npy"
    MaskPath = Paths.RunDirectory / Args.FeatureConfig / "Features" / f"{Args.Split}Masks.npy"
    CheckpointPath = ModelRoot / "Checkpoints" / Args.Checkpoint

    if not CheckpointPath.exists():
        raise FileNotFoundError(CheckpointPath)

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
        pin_memory=torch.cuda.is_available(),
    )

    Checkpoint = torch.load(CheckpointPath, map_location=Device)

    Model = CreateModel(
        ModelName=Checkpoint.get("ModelName", Args.ModelName),
        InputChannels=InputChannels,
        OutputChannels=1,
        ModelParameters=Checkpoint.get("ModelParameters", {}),
    ).to(Device)

    Model.load_state_dict(Checkpoint["ModelStateDict"])
    Model.eval()

    Thresholds = np.linspace(Args.MinThreshold, Args.MaxThreshold, Args.NumThresholds)

    Counts = {
        float(T): {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for T in Thresholds
    }

    with torch.no_grad():
        for Batch in Loader:
            Features = Batch["features"].to(Device)
            Masks = Batch["mask"].to(Device)

            Prob = torch.sigmoid(Model(Features)).detach().cpu().numpy()
            Target = (Masks.detach().cpu().numpy() >= 0.5)

            for Threshold in Thresholds:
                Threshold = float(Threshold)
                Pred = Prob >= Threshold

                Counts[Threshold]["TP"] += int(np.logical_and(Pred, Target).sum())
                Counts[Threshold]["FP"] += int(np.logical_and(Pred, np.logical_not(Target)).sum())
                Counts[Threshold]["FN"] += int(np.logical_and(np.logical_not(Pred), Target).sum())
                Counts[Threshold]["TN"] += int(np.logical_and(np.logical_not(Pred), np.logical_not(Target)).sum())

    Rows = []

    for Threshold, C in Counts.items():
        Metrics = ComputeMetricsFromCounts(
            TP=C["TP"],
            FP=C["FP"],
            FN=C["FN"],
            TN=C["TN"],
        )

        Rows.append(
            {
                "RunTag": Args.RunTag,
                "FeatureConfig": Args.FeatureConfig,
                "ModelName": Args.ModelName,
                "RunName": Args.RunName,
                "ModelRunId": ModelRunId,
                "Split": Args.Split,
                "Checkpoint": Args.Checkpoint,
                "Threshold": Threshold,
                **C,
                **Metrics,
            }
        )

    Output = pd.DataFrame(Rows)
    OutputPath = ModelRoot / "Metrics" / f"{Args.Split}ThresholdSweep.csv"
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    Output.to_csv(OutputPath, index=False)

    Best = Output.sort_values("GlobalDice", ascending=False).iloc[0]

    print("\n=== Threshold sweep completed ===")
    print("Output:", OutputPath)
    print("Best threshold by GlobalDice:")
    print(Best[["Threshold", "GlobalDice", "GlobalIoU", "GlobalPrecision", "GlobalRecall"]].to_string())


if __name__ == "__main__":
    Main()
