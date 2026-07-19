#!/usr/bin/env python3
"""
Step09CheckDataLoaders.py

Verifica DataLoaders PyTorch para tensores precomputados.

Verifica:
- carga de Dataset;
- creación de DataLoader;
- shapes de batches;
- tipos;
- valores finitos;
- máscaras binarias;
- compatibilidad con reflect padding del futuro modelo U-Net.

Importante:
El reflect padding NO se guarda en los tensores.
Se prueba aquí solo como validación de compatibilidad:
B x C x 200 x 200 -> pad reflect -> B x C x 208 x 208.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml
from Source.FeatureTensorDataset import FeatureTensorDataset
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


SPLITS = ["Train", "Validation", "Test"]


def LoadFeatureConfig(ProjectRoot: Path, ConfigName: str) -> dict[str, Any]:
    """Carga ConfigA.yaml / ConfigB.yaml."""
    ConfigPath = ProjectRoot / "Configs" / f"{ConfigName}.yaml"

    if not ConfigPath.exists():
        raise FileNotFoundError(ConfigPath)

    Config = LoadYaml(ConfigPath)

    if "InputChannels" not in Config or "Features" not in Config:
        raise KeyError(f"{ConfigPath} debe contener InputChannels y Features.")

    return Config


def CheckBatch(
    Batch: dict[str, torch.Tensor],
    ExpectedChannels: int,
    ExpectedHeight: int,
    ExpectedWidth: int,
    ReflectPadding: int,
) -> dict[str, Any]:
    """Verifica un batch de DataLoader."""
    Features = Batch["features"]
    Masks = Batch["mask"]

    if Features.ndim != 4:
        raise ValueError(f"Batch features debe ser B x C x H x W. Recibido: {Features.shape}")

    if Masks.ndim != 4:
        raise ValueError(f"Batch mask debe ser B x 1 x H x W. Recibido: {Masks.shape}")

    BatchSize = int(Features.shape[0])

    ShapeOk = tuple(Features.shape[1:]) == (ExpectedChannels, ExpectedHeight, ExpectedWidth)
    MaskShapeOk = tuple(Masks.shape[1:]) == (1, ExpectedHeight, ExpectedWidth)

    if not ShapeOk:
        raise AssertionError(
            f"Feature batch shape esperado (*,{ExpectedChannels},{ExpectedHeight},{ExpectedWidth}), "
            f"recibido {tuple(Features.shape)}"
        )

    if not MaskShapeOk:
        raise AssertionError(
            f"Mask batch shape esperado (*,1,{ExpectedHeight},{ExpectedWidth}), "
            f"recibido {tuple(Masks.shape)}"
        )

    FeatureFinite = bool(torch.isfinite(Features).all().item())
    MaskFinite = bool(torch.isfinite(Masks).all().item())

    UniqueMaskValues = torch.unique(Masks)
    MaskBinary = bool(torch.all((UniqueMaskValues == 0) | (UniqueMaskValues == 1)).item())

    Padded = F.pad(
        Features,
        pad=(ReflectPadding, ReflectPadding, ReflectPadding, ReflectPadding),
        mode="reflect",
    )

    ExpectedPaddedShape = (
        BatchSize,
        ExpectedChannels,
        ExpectedHeight + 2 * ReflectPadding,
        ExpectedWidth + 2 * ReflectPadding,
    )

    ReflectPaddingOk = tuple(Padded.shape) == ExpectedPaddedShape

    Cropped = Padded[
        :,
        :,
        ReflectPadding:-ReflectPadding,
        ReflectPadding:-ReflectPadding,
    ]

    CropBackOk = tuple(Cropped.shape) == tuple(Features.shape)

    return {
        "BatchSize": BatchSize,
        "FeatureBatchShape": list(Features.shape),
        "MaskBatchShape": list(Masks.shape),
        "FeatureDtype": str(Features.dtype),
        "MaskDtype": str(Masks.dtype),
        "FeatureMin": float(Features.min().item()),
        "FeatureMax": float(Features.max().item()),
        "MaskPositivePixels": int((Masks > 0).sum().item()),
        "FeatureFinite": FeatureFinite,
        "MaskFinite": MaskFinite,
        "MaskBinary": MaskBinary,
        "UniqueMaskValues": ",".join(str(float(Value.item())) for Value in UniqueMaskValues),
        "ReflectPadding": ReflectPadding,
        "PaddedShape": list(Padded.shape),
        "ReflectPaddingOk": ReflectPaddingOk,
        "CropBackShape": list(Cropped.shape),
        "CropBackOk": CropBackOk,
        "AllChecksPassed": bool(
            ShapeOk
            and MaskShapeOk
            and FeatureFinite
            and MaskFinite
            and MaskBinary
            and ReflectPaddingOk
            and CropBackOk
        ),
    }


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Verifica DataLoaders PyTorch.")
    Parser = AddCommonArguments(Parser)
    Parser.add_argument(
        "--FeatureConfigs",
        nargs="+",
        default=["ConfigA", "ConfigB"],
        choices=["ConfigA", "ConfigB"],
    )
    Parser.add_argument("--BatchSize", type=int, default=4)
    Parser.add_argument("--NumWorkers", type=int, default=0)
    Parser.add_argument("--BatchesToCheck", type=int, default=2)
    Parser.add_argument("--ExpectedHeight", type=int, default=200)
    Parser.add_argument("--ExpectedWidth", type=int, default=200)
    Parser.add_argument("--ReflectPadding", type=int, default=4)
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    Logger = CreateLogger(
        "Step09CheckDataLoaders",
        Paths.LogsDirectory / "Step09CheckDataLoaders.log",
    )

    Rows = []

    for ConfigName in Args.FeatureConfigs:
        Config = LoadFeatureConfig(Paths.ProjectRoot, ConfigName)
        ExpectedChannels = int(Config["InputChannels"])

        Logger.info("Checking dataloaders for %s", ConfigName)

        for SplitName in SPLITS:
            FeaturePath = Paths.RunDirectory / ConfigName / "Features" / f"{SplitName}Features.npy"
            MaskPath = Paths.RunDirectory / ConfigName / "Features" / f"{SplitName}Masks.npy"

            DatasetObject = FeatureTensorDataset(
                FeaturePath=FeaturePath,
                MaskPath=MaskPath,
                ExpectedChannels=ExpectedChannels,
                ExpectedHeight=Args.ExpectedHeight,
                ExpectedWidth=Args.ExpectedWidth,
            )

            Loader = DataLoader(
                DatasetObject,
                batch_size=Args.BatchSize,
                shuffle=False,
                num_workers=Args.NumWorkers,
                pin_memory=False,
            )

            for BatchIndex, Batch in enumerate(Loader):
                if BatchIndex >= Args.BatchesToCheck:
                    break

                Check = CheckBatch(
                    Batch=Batch,
                    ExpectedChannels=ExpectedChannels,
                    ExpectedHeight=Args.ExpectedHeight,
                    ExpectedWidth=Args.ExpectedWidth,
                    ReflectPadding=Args.ReflectPadding,
                )

                Row = {
                    "FeatureConfig": ConfigName,
                    "Split": SplitName,
                    "DatasetLength": len(DatasetObject),
                    "BatchIndex": int(BatchIndex),
                    "ExpectedChannels": ExpectedChannels,
                    "Features": ",".join(Config["Features"]),
                    **Check,
                }

                Rows.append(Row)

                Logger.info(
                    "%s %s batch %d OK: %s",
                    ConfigName,
                    SplitName,
                    BatchIndex,
                    Row["AllChecksPassed"],
                )

    Result = pd.DataFrame(Rows)

    if Result.empty:
        raise RuntimeError("No se verificó ningún batch.")

    if not bool(Result["AllChecksPassed"].all()):
        Failed = Result[~Result["AllChecksPassed"]]
        raise AssertionError(f"Fallaron DataLoaders:\n{Failed.to_string(index=False)}")

    OutputTables = {}

    for ConfigName in Args.FeatureConfigs:
        ConfigRows = Result[Result["FeatureConfig"] == ConfigName]
        OutputPath = Paths.RunDirectory / ConfigName / "Tables" / "DataLoaderCheck.csv"
        ConfigRows.to_csv(OutputPath, index=False)
        OutputTables[ConfigName] = OutputPath

    AuditPath = Paths.AuditDirectory / "DataLoaderCheckAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    Audit = BuildAuditRecord(
        ScriptName="Step09CheckDataLoaders.py",
        RunTag=Args.RunTag,
        Parameters={
            "FeatureConfigs": Args.FeatureConfigs,
            "BatchSize": Args.BatchSize,
            "NumWorkers": Args.NumWorkers,
            "BatchesToCheck": Args.BatchesToCheck,
            "ExpectedHeight": Args.ExpectedHeight,
            "ExpectedWidth": Args.ExpectedWidth,
            "ReflectPadding": Args.ReflectPadding,
        },
        Inputs={
            ConfigName: {
                SplitName: {
                    "Features": str(
                        Paths.RunDirectory / ConfigName / "Features" / f"{SplitName}Features.npy"
                    ),
                    "Masks": str(
                        Paths.RunDirectory / ConfigName / "Features" / f"{SplitName}Masks.npy"
                    ),
                }
                for SplitName in SPLITS
            }
            for ConfigName in Args.FeatureConfigs
        },
        Outputs={
            **{f"{Config}DataLoaderCheck": str(PathItem) for Config, PathItem in OutputTables.items()},
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "AllChecksPassed": True,
            "CheckedBatches": int(len(Result)),
            "ReflectPaddingTest": "B,C,200,200 -> B,C,208,208 -> crop back to B,C,200,200",
        },
    )

    WriteJson(Audit, AuditPath)

    for Config, PathItem in OutputTables.items():
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step09CheckDataLoaders",
            Config=Config,
            Model="None",
            OutputType="Table",
            RelativePath=str(PathItem.relative_to(Paths.RunDirectory)),
            Created=PathItem.exists(),
            Description=f"Verificación de DataLoader PyTorch para {Config}.",
        )

    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=Args.RunTag,
        Step="Step09CheckDataLoaders",
        Config="Project",
        Model="None",
        OutputType="Audit",
        RelativePath=str(AuditPath.relative_to(Paths.RunDirectory)),
        Created=AuditPath.exists(),
        Description="Auditoría de verificación de DataLoaders.",
    )

    print("\n=== DataLoader check ===")
    print(Result[[
        "FeatureConfig",
        "Split",
        "DatasetLength",
        "BatchIndex",
        "FeatureBatchShape",
        "MaskBatchShape",
        "PaddedShape",
        "AllChecksPassed",
    ]].to_string(index=False))

    print("\nStep09CheckDataLoaders completed successfully.")


if __name__ == "__main__":
    Main()
