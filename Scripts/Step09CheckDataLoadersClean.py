#!/usr/bin/env python3
"""
Step09CheckDataLoadersClean.py

Valida DataLoaders para ConfigB construido con MBMPPlus no supervisado.

Chequea:
- carga de Train/Validation/Test Features.npy y Masks.npy
- Dataset compatible con torch
- batches con shape correcto
- dtype correcto
- valores finitos en X
- máscaras binarias en Y
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


SPLITS = ["Train", "Validation", "Test"]


class NpySegmentationDataset(Dataset):
    def __init__(self, FeaturePath: Path, MaskPath: Path):
        self.X = np.load(FeaturePath, mmap_mode="r")
        self.Y = np.load(MaskPath, mmap_mode="r")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"N distinto: X={self.X.shape}, Y={self.Y.shape}")

        if self.X.ndim != 4:
            raise ValueError(f"X debe ser N,C,H,W. Recibido: {self.X.shape}")

        if self.Y.ndim != 4:
            raise ValueError(f"Y debe ser N,1,H,W. Recibido: {self.Y.shape}")

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, index):
        x = np.asarray(self.X[index], dtype=np.float32)
        y = np.asarray(self.Y[index], dtype=np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(x), torch.from_numpy(y)


def CheckSplit(Paths, Split: str, BatchSize: int, NumWorkers: int) -> dict:
    FeatureDir = Paths.RunDirectory / "ConfigB" / "Features"

    FeaturePath = FeatureDir / f"{Split}Features.npy"
    MaskPath = FeatureDir / f"{Split}Masks.npy"

    if not FeaturePath.exists():
        raise FileNotFoundError(FeaturePath)

    if not MaskPath.exists():
        raise FileNotFoundError(MaskPath)

    DatasetObject = NpySegmentationDataset(FeaturePath, MaskPath)

    Loader = DataLoader(
        DatasetObject,
        batch_size=BatchSize,
        shuffle=False,
        num_workers=NumWorkers,
        pin_memory=False,
    )

    XBatch, YBatch = next(iter(Loader))

    if XBatch.ndim != 4:
        raise AssertionError(f"{Split}: XBatch debe ser B,C,H,W. Recibido {tuple(XBatch.shape)}")

    if YBatch.ndim != 4:
        raise AssertionError(f"{Split}: YBatch debe ser B,1,H,W. Recibido {tuple(YBatch.shape)}")

    if XBatch.shape[1:] != torch.Size([9, 200, 200]):
        raise AssertionError(f"{Split}: XBatch shape inesperado {tuple(XBatch.shape)}")

    if YBatch.shape[1:] != torch.Size([1, 200, 200]):
        raise AssertionError(f"{Split}: YBatch shape inesperado {tuple(YBatch.shape)}")

    if XBatch.dtype != torch.float32:
        raise AssertionError(f"{Split}: XBatch dtype {XBatch.dtype}, esperado torch.float32")

    if YBatch.dtype != torch.float32:
        raise AssertionError(f"{Split}: YBatch dtype {YBatch.dtype}, esperado torch.float32")

    if not torch.isfinite(XBatch).all():
        raise AssertionError(f"{Split}: XBatch contiene NaN/Inf")

    UniqueY = torch.unique(YBatch)
    if not set(UniqueY.cpu().numpy().tolist()).issubset({0.0, 1.0}):
        raise AssertionError(f"{Split}: YBatch no es binario. Valores: {UniqueY}")

    return {
        "Split": Split,
        "Samples": len(DatasetObject),
        "BatchSize": int(BatchSize),
        "XBatchShape": str(tuple(XBatch.shape)),
        "YBatchShape": str(tuple(YBatch.shape)),
        "XBatchDtype": str(XBatch.dtype),
        "YBatchDtype": str(YBatch.dtype),
        "XFinite": bool(torch.isfinite(XBatch).all().item()),
        "YBinary": True,
        "XMin": float(XBatch.min().item()),
        "XMax": float(XBatch.max().item()),
        "YPositivePixelsInBatch": int(YBatch.sum().item()),
    }


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Clean dataloader check for ConfigB.")
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--BatchSize", type=int, default=4)
    Parser.add_argument("--NumWorkers", type=int, default=0)

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)

    Rows = []

    for Split in SPLITS:
        Row = CheckSplit(
            Paths=Paths,
            Split=Split,
            BatchSize=Args.BatchSize,
            NumWorkers=Args.NumWorkers,
        )
        Rows.append(Row)

    Check = pd.DataFrame(Rows)

    TablesDir = Paths.RunDirectory / "ConfigB" / "Tables"
    TablesDir.mkdir(parents=True, exist_ok=True)

    OutPath = TablesDir / "DataLoaderCheck.csv"
    Check.to_csv(OutPath, index=False)

    print("\n=== CLEAN DATALOADER CHECK OK ===")
    print("RunTag:", Args.RunTag)
    print("Output:", OutPath)
    print(Check.to_string(index=False))


if __name__ == "__main__":
    Main()
