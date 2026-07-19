"""
Losses.py

Funciones de pérdida para segmentación binaria.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss calculada sobre logits."""

    def __init__(self, Smooth: float = 1.0) -> None:
        super().__init__()
        self.Smooth = float(Smooth)

    def forward(self, Logits: torch.Tensor, Targets: torch.Tensor) -> torch.Tensor:
        Probabilities = torch.sigmoid(Logits)

        Probabilities = Probabilities.reshape(Probabilities.shape[0], -1)
        Targets = Targets.reshape(Targets.shape[0], -1).float()

        Intersection = (Probabilities * Targets).sum(dim=1)
        Denominator = Probabilities.sum(dim=1) + Targets.sum(dim=1)

        Dice = (2.0 * Intersection + self.Smooth) / (Denominator + self.Smooth)

        return 1.0 - Dice.mean()


class BceDiceLoss(nn.Module):
    """Combinación BCEWithLogitsLoss + DiceLoss."""

    def __init__(
        self,
        BceWeight: float = 0.5,
        DiceWeight: float = 0.5,
        Smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if BceWeight < 0 or DiceWeight < 0:
            raise ValueError("BceWeight y DiceWeight deben ser no negativos.")

        if BceWeight + DiceWeight <= 0:
            raise ValueError("La suma de pesos debe ser positiva.")

        self.BceWeight = float(BceWeight)
        self.DiceWeight = float(DiceWeight)

        self.Bce = nn.BCEWithLogitsLoss()
        self.Dice = DiceLoss(Smooth=Smooth)

    def forward(self, Logits: torch.Tensor, Targets: torch.Tensor) -> torch.Tensor:
        BceValue = self.Bce(Logits, Targets.float())
        DiceValue = self.Dice(Logits, Targets.float())

        return self.BceWeight * BceValue + self.DiceWeight * DiceValue
