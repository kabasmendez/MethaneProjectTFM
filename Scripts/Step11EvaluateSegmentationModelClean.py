#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step11EvaluateSegmentationModelClean.py

Evalúa modelos de segmentación entrenados con ConfigB o ConfigC.

Entradas:
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/<Split>Features.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/<Split>Masks.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Checkpoints/<Checkpoint>

Salidas:
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Metrics/<Split>MetricsSummary.csv
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Metrics/<Split>MetricsBySample.csv
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Tables/<Split>EvaluationSummary.csv

Notas:
- Soporta ConfigB y ConfigC.
- El número de canales se toma de X.shape[1].
- Si el checkpoint contiene input_channels, se valida contra X.shape[1].
- Genera columnas compatibles con Step12:
  SampleId, GroundTruthPixels, PredictedPixels, FalsePositivePixels,
  FalseNegativePixels, Dice, IoU, Precision, Recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Source.Models.ModelFactory import CreateModel


FEATURE_READY_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}


class NpySegmentationEvalDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, sample_ids: list[str] | None = None):
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
            raise ValueError(f"N distinto entre X e Y: X={self.x.shape}, Y={self.y.shape}")
        if self.y.shape[1] != 1:
            raise ValueError(f"Y debe tener 1 canal. Recibido: {self.y.shape}")
        if self.x.shape[2:] != self.y.shape[2:]:
            raise ValueError(f"H,W distinto: X={self.x.shape}, Y={self.y.shape}")

        if sample_ids is None:
            sample_ids = [str(i) for i in range(self.x.shape[0])]

        if len(sample_ids) != self.x.shape[0]:
            raise ValueError(
                f"sample_ids tiene {len(sample_ids)} elementos, "
                f"pero X tiene {self.x.shape[0]} muestras."
            )

        self.sample_ids = [str(x) for x in sample_ids]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.x[idx], dtype=np.float32, copy=True))
        y = torch.from_numpy(np.array(self.y[idx], dtype=np.float32, copy=True))
        sample_id = self.sample_ids[idx]
        return x, y, sample_id


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_sample_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SampleId" in df.columns:
        df["SampleId"] = df["SampleId"].astype(str)
        return df

    for candidate in ["sample_id", "id", "Id", "ID", "sampleId", "SampleID"]:
        if candidate in df.columns:
            df["SampleId"] = df[candidate].astype(str)
            return df

    raise KeyError(f"No encontré columna SampleId/id. Columnas: {list(df.columns)}")


def load_split_sample_ids(run_root: Path, split: str, expected_n: int) -> list[str]:
    path = run_root / "Tables" / FEATURE_READY_FILES[split]

    if not path.exists():
        print(f"WARNING: no existe {path}. Usaré índices 0..N-1 como SampleId.")
        return [str(i) for i in range(expected_n)]

    df = pd.read_csv(path)
    df = ensure_sample_id_column(df)

    sample_ids = df["SampleId"].astype(str).tolist()

    if len(sample_ids) != expected_n:
        print(
            f"WARNING: split table tiene {len(sample_ids)} ids, "
            f"pero features tienen {expected_n}. Ajustando."
        )

        if len(sample_ids) > expected_n:
            sample_ids = sample_ids[:expected_n]
        else:
            sample_ids = sample_ids + [str(i) for i in range(len(sample_ids), expected_n)]

    return sample_ids


def extract_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        return checkpoint

    raise TypeError(f"Checkpoint no es dict. Tipo: {type(checkpoint)}")


def get_state_dict_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ["model_state_dict", "state_dict", "model", "model_state"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # Fallback: algunos checkpoints son directamente state_dict-like.
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]

    raise KeyError(f"No encontré state dict en checkpoint. Llaves: {list(checkpoint.keys())}")


def get_model_parameters(checkpoint: dict[str, Any], training_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(checkpoint.get("model_parameters"), dict):
        return dict(checkpoint["model_parameters"])

    if isinstance(checkpoint.get("ModelParameters"), dict):
        return dict(checkpoint["ModelParameters"])

    if isinstance(training_config.get("ModelParameters"), dict):
        return dict(training_config["ModelParameters"])

    # Defaults compatibles con los entrenamientos usados.
    return {
        "BaseChannels": int(training_config.get("BaseChannels", 32)),
        "Dropout": float(training_config.get("Dropout", 0.05)),
        "TransformerLayers": int(training_config.get("TransformerLayers", 2)),
        "TransformerHeads": int(training_config.get("TransformerHeads", 4)),
        "UseSqueezeExcitation": bool(training_config.get("UseSqueezeExcitation", False)),
        "ReflectPadding": int(training_config.get("ReflectPadding", 4)),
    }


def build_model(
    *,
    model_name: str,
    input_channels: int,
    output_channels: int,
    model_parameters: dict[str, Any],
) -> torch.nn.Module:
    return CreateModel(
        ModelName=model_name,
        InputChannels=input_channels,
        OutputChannels=output_channels,
        ModelParameters=model_parameters,
    )


def compute_sample_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float | int]:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    tp = int(np.logical_and(pred == 1, target == 1).sum())
    fp = int(np.logical_and(pred == 1, target == 0).sum())
    fn = int(np.logical_and(pred == 0, target == 1).sum())
    tn = int(np.logical_and(pred == 0, target == 0).sum())

    gt_pixels = int(target.sum())
    pred_pixels = int(pred.sum())

    dice = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
    iou = float((tp + eps) / (tp + fp + fn + eps))
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))

    return {
        "GroundTruthPixels": gt_pixels,
        "PredictedPixels": pred_pixels,
        "TruePositivePixels": tp,
        "FalsePositivePixels": fp,
        "FalseNegativePixels": fn,
        "TrueNegativePixels": tn,
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
    }


@torch.no_grad()
def evaluate_model(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> pd.DataFrame:
    model.eval()

    rows: list[dict[str, Any]] = []

    for x, y, sample_ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(torch.uint8)

        preds_np = preds.detach().cpu().numpy()
        y_np = (y.detach().cpu().numpy() > 0.5).astype(np.uint8)
        probs_np = probs.detach().cpu().numpy()

        for i, sample_id in enumerate(sample_ids):
            pred_mask = preds_np[i, 0]
            gt_mask = y_np[i, 0]
            prob_map = probs_np[i, 0]

            metrics = compute_sample_metrics(pred_mask, gt_mask)

            row = {
                "SampleId": str(sample_id),
                "Threshold": float(threshold),
                "PredictionProbabilityMean": float(np.mean(prob_map)),
                "PredictionProbabilityMax": float(np.max(prob_map)),
                **metrics,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def build_summary(
    *,
    run_tag: str,
    feature_config: str,
    model_name: str,
    run_name: str,
    run_id: str,
    split: str,
    checkpoint_name: str,
    threshold: float,
    input_channels: int,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    tp = int(sample_df["TruePositivePixels"].sum())
    fp = int(sample_df["FalsePositivePixels"].sum())
    fn = int(sample_df["FalseNegativePixels"].sum())

    eps = 1e-6

    global_dice = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
    global_iou = float((tp + eps) / (tp + fp + fn + eps))
    global_precision = float((tp + eps) / (tp + fp + eps))
    global_recall = float((tp + eps) / (tp + fn + eps))

    row = {
        "RunTag": run_tag,
        "FeatureConfig": feature_config,
        "ModelName": model_name,
        "RunName": run_name,
        "RunId": run_id,
        "ModelRunId": run_id,
        "Split": split,
        "Checkpoint": checkpoint_name,
        "Threshold": float(threshold),
        "Samples": int(len(sample_df)),
        "InputChannels": int(input_channels),
        "MeanDice": float(sample_df["Dice"].mean()),
        "MeanIoU": float(sample_df["IoU"].mean()),
        "MeanPrecision": float(sample_df["Precision"].mean()),
        "MeanRecall": float(sample_df["Recall"].mean()),
        "GlobalDice": global_dice,
        "GlobalIoU": global_iou,
        "GlobalPrecision": global_precision,
        "GlobalRecall": global_recall,
        "GroundTruthPixels": int(sample_df["GroundTruthPixels"].sum()),
        "PredictedPixels": int(sample_df["PredictedPixels"].sum()),
        "TruePositivePixels": tp,
        "FalsePositivePixels": fp,
        "FalseNegativePixels": fn,
    }

    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa modelo de segmentación para ConfigB o ConfigC."
    )

    parser.add_argument("--RunTag", required=True)
    parser.add_argument("--FeatureConfig", default="ConfigB", choices=["ConfigB", "ConfigC"])
    parser.add_argument("--ModelName", required=True)
    parser.add_argument("--RunName", required=True)
    parser.add_argument("--Split", default="Test", choices=["Train", "Validation", "Test"])
    parser.add_argument("--Checkpoint", default="BestModel.pt")
    parser.add_argument("--BatchSize", type=int, default=8)
    parser.add_argument("--Threshold", type=float, default=0.5)
    parser.add_argument("--Device", default="auto")
    parser.add_argument("--NumWorkers", type=int, default=0)

    args = parser.parse_args()

    device = get_device(args.Device)

    run_id = f"{args.ModelName}_{args.RunName}"

    run_root = PROJECT_ROOT / "Outputs" / "Experiments" / args.RunTag
    feature_dir = run_root / args.FeatureConfig / "Features"
    model_root = run_root / args.FeatureConfig / run_id

    checkpoint_path = model_root / "Checkpoints" / args.Checkpoint
    training_config_path = model_root / "TrainingConfig.json"

    metrics_dir = model_root / "Metrics"
    tables_dir = model_root / "Tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    x_path = feature_dir / f"{args.Split}Features.npy"
    y_path = feature_dir / f"{args.Split}Masks.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"No existe features para evaluar: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"No existe masks para evaluar: {y_path}")

    x_mem = np.load(x_path, mmap_mode="r")
    input_channels = int(x_mem.shape[1])
    output_channels = 1
    sample_ids = load_split_sample_ids(run_root, args.Split, expected_n=int(x_mem.shape[0]))

    dataset = NpySegmentationEvalDataset(x_path, y_path, sample_ids=sample_ids)

    loader = DataLoader(
        dataset,
        batch_size=int(args.BatchSize),
        shuffle=False,
        num_workers=int(args.NumWorkers),
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = extract_checkpoint_payload(checkpoint_path, device=device)
    training_config = load_json_if_exists(training_config_path)

    checkpoint_input_channels = checkpoint.get("input_channels", checkpoint.get("InputChannels", None))

    if checkpoint_input_channels is not None:
        checkpoint_input_channels = int(checkpoint_input_channels)
        if checkpoint_input_channels != input_channels:
            raise ValueError(
                f"Checkpoint input_channels={checkpoint_input_channels}, "
                f"pero X tiene {input_channels} canales."
            )

    model_parameters = get_model_parameters(checkpoint, training_config)

    model = build_model(
        model_name=args.ModelName,
        input_channels=input_channels,
        output_channels=output_channels,
        model_parameters=model_parameters,
    )

    state_dict = get_state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    print("")
    print("=== CLEAN EVALUATION START ===")
    print(f"RunTag: {args.RunTag}")
    print(f"FeatureConfig: {args.FeatureConfig}")
    print(f"RunId: {run_id}")
    print(f"Split: {args.Split}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Samples: {len(dataset)}")
    print(f"InputChannels: {input_channels}")
    print(f"Threshold: {args.Threshold}")

    by_sample = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        threshold=float(args.Threshold),
    )

    summary = build_summary(
        run_tag=args.RunTag,
        feature_config=args.FeatureConfig,
        model_name=args.ModelName,
        run_name=args.RunName,
        run_id=run_id,
        split=args.Split,
        checkpoint_name=args.Checkpoint,
        threshold=float(args.Threshold),
        input_channels=input_channels,
        sample_df=by_sample,
    )

    # Nombres compatibles con Step12/Step15.
    metrics_summary_path = metrics_dir / f"{args.Split}MetricsSummary.csv"
    metrics_by_sample_path = metrics_dir / f"{args.Split}MetricsBySample.csv"
    eval_summary_path = tables_dir / f"{args.Split}EvaluationSummary.csv"

    summary.to_csv(metrics_summary_path, index=False)
    by_sample.to_csv(metrics_by_sample_path, index=False)
    summary.to_csv(eval_summary_path, index=False)

    print("")
    print("=== CLEAN EVALUATION COMPLETED ===")
    print(f"Summary: {metrics_summary_path}")
    print(f"By sample: {metrics_by_sample_path}")
    print(f"Evaluation table: {eval_summary_path}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
