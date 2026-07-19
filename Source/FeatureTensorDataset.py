"""
FeatureTensorDataset.py

Dataset PyTorch para tensores precomputados de MethaneProjectTFM.

Lee:
- <Split>Features.npy: N x C x 200 x 200
- <Split>Masks.npy:    N x 1 x 200 x 200

No aplica reflect padding. El reflect padding pertenece al modelo U-Net.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureTensorDataset(Dataset):
    """Dataset PyTorch basado en tensores .npy precomputados."""

    def __init__(
        self,
        FeaturePath: str | Path,
        MaskPath: str | Path,
        ExpectedChannels: int | None = None,
        ExpectedHeight: int = 200,
        ExpectedWidth: int = 200,
    ) -> None:
        self.FeaturePath = Path(FeaturePath)
        self.MaskPath = Path(MaskPath)

        if not self.FeaturePath.exists():
            raise FileNotFoundError(f"No existe FeaturePath: {self.FeaturePath}")

        if not self.MaskPath.exists():
            raise FileNotFoundError(f"No existe MaskPath: {self.MaskPath}")

        self.Features = np.load(self.FeaturePath, mmap_mode="r")
        self.Masks = np.load(self.MaskPath, mmap_mode="r")

        if self.Features.ndim != 4:
            raise ValueError(f"Features debe ser N x C x H x W. Recibido: {self.Features.shape}")

        if self.Masks.ndim != 4:
            raise ValueError(f"Masks debe ser N x 1 x H x W. Recibido: {self.Masks.shape}")

        if self.Features.shape[0] != self.Masks.shape[0]:
            raise ValueError(
                f"Features y Masks tienen N distinto: {self.Features.shape[0]} vs {self.Masks.shape[0]}"
            )

        if self.Masks.shape[1] != 1:
            raise ValueError(f"Masks debe tener canal 1. Recibido: {self.Masks.shape}")

        if ExpectedChannels is not None and self.Features.shape[1] != ExpectedChannels:
            raise ValueError(
                f"Channels esperado {ExpectedChannels}, recibido {self.Features.shape[1]}"
            )

        if self.Features.shape[2:] != (ExpectedHeight, ExpectedWidth):
            raise ValueError(
                f"Features H/W esperado {(ExpectedHeight, ExpectedWidth)}, "
                f"recibido {self.Features.shape[2:]}"
            )

        if self.Masks.shape[2:] != (ExpectedHeight, ExpectedWidth):
            raise ValueError(
                f"Masks H/W esperado {(ExpectedHeight, ExpectedWidth)}, "
                f"recibido {self.Masks.shape[2:]}"
            )

    def __len__(self) -> int:
        return int(self.Features.shape[0])

    def __getitem__(self, Index: int) -> dict[str, Any]:
        if Index < 0 or Index >= len(self):
            raise IndexError(f"Index fuera de rango: {Index}")

        # np.array(copy=True) evita warnings de PyTorch sobre memmaps read-only.
        Feature = torch.from_numpy(np.array(self.Features[Index], dtype=np.float32, copy=True))
        Mask = torch.from_numpy(np.array(self.Masks[Index], dtype=np.float32, copy=True))

        return {
            "features": Feature,
            "mask": Mask,
            "index": int(Index),
        }
