#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step10TrainSegmentationModelClean.py

Entrena modelos de segmentación para ConfigB o ConfigC.

Soporta:
- ConfigB: 9 canales
- ConfigC: 12 canales, ConfigB + viento

Entradas:
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/TrainFeatures.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/TrainMasks.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/ValidationFeatures.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/ValidationMasks.npy

Salidas:
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Checkpoints/BestModel.pt
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Checkpoints/LastModel.pt
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Metrics/TrainingHistory.csv
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Metrics/BestEpochSummary.csv
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/TrainingConfig.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Source.Models.ModelFactory import CreateModel


class NpySegmentationDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path):
        if not x_path.exists():
            raise FileNotFoundError(f"No existe X: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"No existe Y: {y_path}")

        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        if self.x.ndim != 4:
            raise ValueError(f"X debe ser N,C,H,W. Recibido: {self.x.shape}")
        if self.y.ndim != 4:
            raise ValueError(f"Y debe ser N,1,H,W. Recibido: {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"N distinto: X={self.x.shape}, Y={self.y.shape}")
        if self.y.shape[1] != 1:
            raise ValueError(f"Y debe tener un canal. Recibido: {self.y.shape}")
        if self.x.shape[2:] != self.y.shape[2:]:
            raise ValueError(f"H,W distinto: X={self.x.shape}, Y={self.y.shape}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.asarray(self.x[idx], dtype=np.float32))
        y = torch.from_numpy(np.asarray(self.y[idx], dtype=np.float32))
        return x, y


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))

        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1.0 - dice.mean()

        return bce_loss + self.dice_weight * dice_loss


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()

    dims = tuple(range(1, preds.ndim))

    tp = torch.sum(preds * targets, dim=dims)
    fp = torch.sum(preds * (1.0 - targets), dim=dims)
    fn = torch.sum((1.0 - preds) * targets, dim=dims)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    global_tp = tp.sum()
    global_fp = fp.sum()
    global_fn = fn.sum()

    global_dice = (2.0 * global_tp + eps) / (2.0 * global_tp + global_fp + global_fn + eps)
    global_iou = (global_tp + eps) / (global_tp + global_fp + global_fn + eps)

    return {
        "MeanDice": float(dice.mean().detach().cpu()),
        "MeanIoU": float(iou.mean().detach().cpu()),
        "GlobalDice": float(global_dice.detach().cpu()),
        "GlobalIoU": float(global_iou.detach().cpu()),
    }


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.train()

    losses = []
    metrics = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        metrics.append(compute_batch_metrics(logits.detach(), y.detach(), threshold=threshold))

    out = average_metric_dicts(metrics)
    out["Loss"] = float(np.mean(losses))
    return out


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()

    losses = []
    metrics = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        loss = criterion(logits, y)

        losses.append(float(loss.detach().cpu()))
        metrics.append(compute_batch_metrics(logits.detach(), y.detach(), threshold=threshold))

    out = average_metric_dicts(metrics)
    out["Loss"] = float(np.mean(losses))
    return out


def build_model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    params = {
        "BaseChannels": int(args.BaseChannels),
        "Dropout": float(args.Dropout),
    }

    if args.ModelName == "TransformerUNet":
        params.update(
            {
                "TransformerLayers": int(args.TransformerLayers),
                "TransformerHeads": int(args.TransformerHeads),
                "UseSqueezeExcitation": bool(args.UseSqueezeExcitation),
                "ReflectPadding": int(args.ReflectPadding),
            }
        )

    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena modelo de segmentación para ConfigB o ConfigC."
    )

    parser.add_argument("--RunTag", required=True)
    parser.add_argument("--FeatureConfig", default="ConfigB", choices=["ConfigB", "ConfigC"])
    parser.add_argument("--ModelName", required=True)
    parser.add_argument("--RunName", required=True)

    parser.add_argument("--Epochs", type=int, default=30)
    parser.add_argument("--BatchSize", type=int, default=4)
    parser.add_argument("--LearningRate", type=float, default=1e-4)
    parser.add_argument("--WeightDecay", type=float, default=1e-5)
    parser.add_argument("--Dropout", type=float, default=0.05)
    parser.add_argument("--BaseChannels", type=int, default=32)
    parser.add_argument("--DiceWeight", type=float, default=1.0)
    parser.add_argument("--Threshold", type=float, default=0.5)

    parser.add_argument("--TransformerLayers", type=int, default=2)
    parser.add_argument("--TransformerHeads", type=int, default=4)
    parser.add_argument("--UseSqueezeExcitation", action="store_true")
    parser.add_argument("--ReflectPadding", type=int, default=4)

    parser.add_argument("--Device", default="auto")
    parser.add_argument("--Seed", type=int, default=42)
    parser.add_argument("--NumWorkers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.Seed)

    device = get_device(args.Device)

    run_root = PROJECT_ROOT / "Outputs" / "Experiments" / args.RunTag
    feature_dir = run_root / args.FeatureConfig / "Features"

    train_x_path = feature_dir / "TrainFeatures.npy"
    train_y_path = feature_dir / "TrainMasks.npy"
    val_x_path = feature_dir / "ValidationFeatures.npy"
    val_y_path = feature_dir / "ValidationMasks.npy"

    train_ds = NpySegmentationDataset(train_x_path, train_y_path)
    val_ds = NpySegmentationDataset(val_x_path, val_y_path)

    input_channels = int(train_ds.x.shape[1])
    output_channels = 1

    if int(val_ds.x.shape[1]) != input_channels:
        raise ValueError(
            f"Train input channels={input_channels}, "
            f"Validation input channels={val_ds.x.shape[1]}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.BatchSize,
        shuffle=True,
        num_workers=args.NumWorkers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.BatchSize,
        shuffle=False,
        num_workers=args.NumWorkers,
        pin_memory=torch.cuda.is_available(),
    )

    model_params = build_model_parameters(args)

    model = CreateModel(
        ModelName=args.ModelName,
        InputChannels=input_channels,
        OutputChannels=output_channels,
        ModelParameters=model_params,
    )

    model = model.to(device)

    criterion = DiceBCELoss(dice_weight=args.DiceWeight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.LearningRate,
        weight_decay=args.WeightDecay,
    )

    run_id = f"{args.ModelName}_{args.RunName}"
    model_root = run_root / args.FeatureConfig / run_id
    checkpoint_dir = model_root / "Checkpoints"
    metrics_dir = model_root / "Metrics"
    tables_dir = model_root / "Tables"

    ensure_dir(checkpoint_dir)
    ensure_dir(metrics_dir)
    ensure_dir(tables_dir)

    training_config = {
        "RunTag": args.RunTag,
        "FeatureConfig": args.FeatureConfig,
        "ModelName": args.ModelName,
        "RunName": args.RunName,
        "RunId": run_id,
        "InputChannels": input_channels,
        "OutputChannels": output_channels,
        "ModelParameters": model_params,
        "Epochs": int(args.Epochs),
        "BatchSize": int(args.BatchSize),
        "LearningRate": float(args.LearningRate),
        "WeightDecay": float(args.WeightDecay),
        "Dropout": float(args.Dropout),
        "BaseChannels": int(args.BaseChannels),
        "DiceWeight": float(args.DiceWeight),
        "Threshold": float(args.Threshold),
        "Device": str(device),
        "Seed": int(args.Seed),
        "TrainSamples": len(train_ds),
        "ValidationSamples": len(val_ds),
        "TrainXPath": str(train_x_path),
        "TrainYPath": str(train_y_path),
        "ValidationXPath": str(val_x_path),
        "ValidationYPath": str(val_y_path),
    }

    save_json(model_root / "TrainingConfig.json", training_config)

    print("")
    print("=== CLEAN TRAINING START ===")
    print(f"RunTag: {args.RunTag}")
    print(f"FeatureConfig: {args.FeatureConfig}")
    print(f"ModelName: {args.ModelName}")
    print(f"RunName: {args.RunName}")
    print(f"RunId: {run_id}")
    print(f"Device: {device}")
    print(f"InputChannels: {input_channels}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    best_val_dice = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, int(args.Epochs) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=float(args.Threshold),
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=float(args.Threshold),
        )

        row = {
            "Epoch": epoch,
            "TrainLoss": train_metrics["Loss"],
            "TrainDice": train_metrics["MeanDice"],
            "TrainIoU": train_metrics["MeanIoU"],
            "TrainGlobalDice": train_metrics["GlobalDice"],
            "TrainGlobalIoU": train_metrics["GlobalIoU"],
            "ValLoss": val_metrics["Loss"],
            "ValDice": val_metrics["MeanDice"],
            "ValIoU": val_metrics["MeanIoU"],
            "ValGlobalDice": val_metrics["GlobalDice"],
            "ValGlobalIoU": val_metrics["GlobalIoU"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.Epochs} | "
            f"TrainDice={row['TrainDice']:.4f} | "
            f"ValDice={row['ValDice']:.4f} | "
            f"ValIoU={row['ValIoU']:.4f} | "
            f"ValLoss={row['ValLoss']:.4f}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_name": args.ModelName,
            "run_name": args.RunName,
            "run_id": run_id,
            "feature_config": args.FeatureConfig,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "model_parameters": model_params,
            "threshold": float(args.Threshold),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
            "training_config": training_config,
        }

        torch.save(checkpoint_payload, checkpoint_dir / "LastModel.pt")

        if row["ValDice"] > best_val_dice:
            best_val_dice = float(row["ValDice"])
            best_epoch = int(epoch)
            torch.save(checkpoint_payload, checkpoint_dir / "BestModel.pt")

    history_df = pd.DataFrame(history)
    history_path = metrics_dir / "TrainingHistory.csv"
    history_df.to_csv(history_path, index=False)

    best_row = history_df.loc[history_df["Epoch"] == best_epoch].copy()
    best_summary_path = metrics_dir / "BestEpochSummary.csv"
    best_row.to_csv(best_summary_path, index=False)

    run_summary = pd.DataFrame(
        [
            {
                "RunTag": args.RunTag,
                "FeatureConfig": args.FeatureConfig,
                "ModelName": args.ModelName,
                "RunName": args.RunName,
                "RunId": run_id,
                "InputChannels": input_channels,
                "Epochs": int(args.Epochs),
                "BestEpoch": best_epoch,
                "BestValMeanDice": best_val_dice,
                "BestModelPath": str(checkpoint_dir / "BestModel.pt"),
                "LastModelPath": str(checkpoint_dir / "LastModel.pt"),
                "TrainingHistoryPath": str(history_path),
            }
        ]
    )

    run_summary_path = tables_dir / "ModelRunSummary.csv"
    run_summary.to_csv(run_summary_path, index=False)

    print("")
    print("=== CLEAN TRAINING COMPLETED ===")
    print(f"RunId: {run_id}")
    print(f"BestEpoch: {best_epoch}")
    print(f"BestValMeanDice: {best_val_dice}")
    print(f"History: {history_path}")
    print(f"Best model: {checkpoint_dir / 'BestModel.pt'}")
    print(f"Run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
