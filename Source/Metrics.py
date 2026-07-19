"""
Metrics.py

Métricas de segmentación binaria por batch.
"""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def ComputeBinarySegmentationMetrics(
    Logits: torch.Tensor,
    Targets: torch.Tensor,
    Threshold: float = 0.5,
    Epsilon: float = 1e-7,
) -> dict[str, Any]:
    """Calcula métricas binarias desde logits."""
    Probabilities = torch.sigmoid(Logits)
    Predictions = Probabilities >= Threshold
    TargetsBool = Targets >= 0.5

    PredictionsFlat = Predictions.reshape(Predictions.shape[0], -1)
    TargetsFlat = TargetsBool.reshape(TargetsBool.shape[0], -1)

    TP = (PredictionsFlat & TargetsFlat).sum(dim=1).float()
    FP = (PredictionsFlat & ~TargetsFlat).sum(dim=1).float()
    FN = (~PredictionsFlat & TargetsFlat).sum(dim=1).float()
    TN = (~PredictionsFlat & ~TargetsFlat).sum(dim=1).float()

    Dice = (2.0 * TP + Epsilon) / (2.0 * TP + FP + FN + Epsilon)
    IoU = (TP + Epsilon) / (TP + FP + FN + Epsilon)
    Precision = (TP + Epsilon) / (TP + FP + Epsilon)
    Recall = (TP + Epsilon) / (TP + FN + Epsilon)

    return {
        "MeanDice": float(Dice.mean().item()),
        "MeanIoU": float(IoU.mean().item()),
        "MeanPrecision": float(Precision.mean().item()),
        "MeanRecall": float(Recall.mean().item()),
        "TP": int(TP.sum().item()),
        "FP": int(FP.sum().item()),
        "FN": int(FN.sum().item()),
        "TN": int(TN.sum().item()),
    }


def AggregateMetricRows(Rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega métricas de múltiples batches."""
    if not Rows:
        raise ValueError("No hay métricas para agregar.")

    TotalTP = int(sum(Row["TP"] for Row in Rows))
    TotalFP = int(sum(Row["FP"] for Row in Rows))
    TotalFN = int(sum(Row["FN"] for Row in Rows))
    TotalTN = int(sum(Row["TN"] for Row in Rows))

    Epsilon = 1e-7

    GlobalDice = (2.0 * TotalTP + Epsilon) / (
        2.0 * TotalTP + TotalFP + TotalFN + Epsilon
    )
    GlobalIoU = (TotalTP + Epsilon) / (
        TotalTP + TotalFP + TotalFN + Epsilon
    )

    return {
        "MeanLoss": float(sum(Row["Loss"] for Row in Rows) / len(Rows)),
        "MeanDice": float(sum(Row["MeanDice"] for Row in Rows) / len(Rows)),
        "MeanIoU": float(sum(Row["MeanIoU"] for Row in Rows) / len(Rows)),
        "MeanPrecision": float(sum(Row["MeanPrecision"] for Row in Rows) / len(Rows)),
        "MeanRecall": float(sum(Row["MeanRecall"] for Row in Rows) / len(Rows)),
        "GlobalDice": float(GlobalDice),
        "GlobalIoU": float(GlobalIoU),
        "TP": TotalTP,
        "FP": TotalFP,
        "FN": TotalFN,
        "TN": TotalTN,
        "Batches": int(len(Rows)),
    }
