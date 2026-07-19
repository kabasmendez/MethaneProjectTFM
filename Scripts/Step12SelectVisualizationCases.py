#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step12SelectVisualizationCases.py

Selecciona casos cualitativos para visualización.

Compatible con ConfigB y ConfigC.

Entradas:
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Metrics/<Split>MetricsBySample.csv
- Outputs/Experiments/<RunTag>/Tables/Split<Split>FeatureReady.csv

Salidas:
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Tables/VisualizationCaseSet_<RunId>.csv
- Outputs/Experiments/<RunTag>/Tables/VisualizationCaseSet.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}


def ensure_sample_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SampleId" in df.columns:
        df["SampleId"] = df["SampleId"].astype(str)
        return df

    for candidate in ["sample_id", "id", "Id", "ID", "sampleId", "SampleID"]:
        if candidate in df.columns:
            df["SampleId"] = df[candidate].astype(str)
            return df

    raise KeyError(f"No encontré SampleId/id. Columnas disponibles: {list(df.columns)}")


def load_split_table(run_root: Path, split: str) -> pd.DataFrame:
    path = run_root / "Tables" / SPLIT_FILES[split]

    if not path.exists():
        raise FileNotFoundError(f"No existe split table: {path}")

    df = pd.read_csv(path)
    df = ensure_sample_id_column(df)
    df["SplitOrder"] = np.arange(len(df), dtype=int)

    return df[["SampleId", "SplitOrder"]].copy()


def load_metrics(model_root: Path, split: str, split_table: pd.DataFrame) -> pd.DataFrame:
    metrics_path = model_root / "Metrics" / f"{split}MetricsBySample.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No existe {metrics_path}. Ejecuta primero Step11 para este modelo."
        )

    df = pd.read_csv(metrics_path)

    if "SampleId" not in df.columns:
        if len(df) != len(split_table):
            raise ValueError(
                f"{metrics_path} no tiene SampleId y len={len(df)}, "
                f"pero split_table len={len(split_table)}."
            )
        df["SampleId"] = split_table["SampleId"].values

    df["SampleId"] = df["SampleId"].astype(str)
    df = df.merge(split_table, on="SampleId", how="left")

    if df["SplitOrder"].isna().any():
        missing = df.loc[df["SplitOrder"].isna(), "SampleId"].head(10).tolist()
        raise ValueError(f"Hay SampleId en métricas que no están en split table: {missing}")

    df["SplitOrder"] = df["SplitOrder"].astype(int)

    required = [
        "GroundTruthPixels",
        "PredictedPixels",
        "FalsePositivePixels",
        "FalseNegativePixels",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
    ]

    for col in required:
        if col not in df.columns:
            raise KeyError(f"Falta columna requerida en métricas: {col}")

    df["TotalErrorPixels"] = df["FalsePositivePixels"] + df["FalseNegativePixels"]
    df["HasGroundTruth"] = df["GroundTruthPixels"] > 0
    df["HasPrediction"] = df["PredictedPixels"] > 0

    return df


def take_unique(df: pd.DataFrame, n: int, used: set[str]) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        sid = str(row["SampleId"])

        if sid in used:
            continue

        rows.append(row)
        used.add(sid)

        if len(rows) >= n:
            break

    if not rows:
        return df.iloc[0:0].copy()

    return pd.DataFrame(rows)


def select_fixed_cases(metrics: pd.DataFrame, n: int, used: set[str]) -> pd.DataFrame:
    positive = metrics[metrics["GroundTruthPixels"] > 0].copy()

    if len(positive) == 0:
        return metrics.iloc[0:0].copy()

    positive = positive.sort_values("GroundTruthPixels").copy()

    if len(positive) >= 3:
        positive["PlumeSizeGroup"] = pd.qcut(
            positive["GroundTruthPixels"].rank(method="first"),
            q=3,
            labels=False,
            duplicates="drop",
        )
    else:
        positive["PlumeSizeGroup"] = 0

    groups = sorted(positive["PlumeSizeGroup"].dropna().unique().tolist())
    per_group = max(1, int(np.ceil(n / max(1, len(groups)))))

    selected_parts = []

    for group in groups:
        gdf = positive[positive["PlumeSizeGroup"] == group].copy()
        gdf = gdf.sort_values(["GroundTruthPixels", "SplitOrder"], ascending=[True, True])
        selected_parts.append(take_unique(gdf, per_group, used))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else metrics.iloc[0:0].copy()

    if len(selected) < n:
        extra = take_unique(positive.sort_values("SplitOrder"), n - len(selected), used)
        selected = pd.concat([selected, extra], ignore_index=True)

    return selected.head(n).copy()


def annotate_cases(df: pd.DataFrame, case_group: str, reason: str) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    out = df.copy()
    out.insert(0, "CaseGroup", case_group)
    out.insert(1, "Order", np.arange(1, len(out) + 1, dtype=int))
    out.insert(2, "SelectionReason", reason)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Selecciona casos para visualización cualitativa.")

    parser.add_argument("--RunTag", required=True)
    parser.add_argument("--FeatureConfig", default="ConfigB", choices=["ConfigB", "ConfigC"])
    parser.add_argument("--ModelName", required=True)
    parser.add_argument("--RunName", required=True)
    parser.add_argument("--Split", default="Test", choices=["Train", "Validation", "Test"])

    parser.add_argument("--FixedCount", type=int, default=9)
    parser.add_argument("--BestCount", type=int, default=3)
    parser.add_argument("--WorstCount", type=int, default=3)
    parser.add_argument("--ErrorCount", type=int, default=3)
    parser.add_argument("--AllowDuplicates", action="store_true")

    args = parser.parse_args()

    run_id = f"{args.ModelName}_{args.RunName}"

    run_root = PROJECT_ROOT / "Outputs" / "Experiments" / args.RunTag
    model_root = run_root / args.FeatureConfig / run_id

    model_tables_dir = model_root / "Tables"
    model_tables_dir.mkdir(parents=True, exist_ok=True)

    root_tables_dir = run_root / "Tables"
    root_tables_dir.mkdir(parents=True, exist_ok=True)

    split_table = load_split_table(run_root, args.Split)
    metrics = load_metrics(model_root, args.Split, split_table)

    used: set[str] = set()

    fixed = select_fixed_cases(metrics, args.FixedCount, used if not args.AllowDuplicates else set())
    fixed = annotate_cases(fixed, "FixedComparisonCases", "PlumeSizeStratified")

    best_candidates = metrics[metrics["HasGroundTruth"]].sort_values(
        ["Dice", "IoU", "GroundTruthPixels"],
        ascending=[False, False, False],
    )
    best = take_unique(best_candidates, args.BestCount, used if not args.AllowDuplicates else set())
    best = annotate_cases(best, "BestPredictions", "HighestDice")

    worst_candidates = metrics[metrics["HasGroundTruth"]].sort_values(
        ["Dice", "IoU", "GroundTruthPixels"],
        ascending=[True, True, False],
    )
    worst = take_unique(worst_candidates, args.WorstCount, used if not args.AllowDuplicates else set())
    worst = annotate_cases(worst, "WorstPredictions", "LowestDice")

    error_candidates = metrics[metrics["HasGroundTruth"]].sort_values(
        ["TotalErrorPixels", "FalsePositivePixels", "FalseNegativePixels"],
        ascending=[False, False, False],
    )
    error = take_unique(error_candidates, args.ErrorCount, used if not args.AllowDuplicates else set())
    error = annotate_cases(error, "ErrorCases", "LargestFalsePositivePlusFalseNegative")

    case_set = pd.concat([fixed, best, worst, error], ignore_index=True)

    if len(case_set) == 0:
        raise ValueError("No se seleccionaron casos.")

    case_set.insert(0, "RunTag", args.RunTag)
    case_set.insert(1, "FeatureConfig", args.FeatureConfig)
    case_set.insert(2, "ModelName", args.ModelName)
    case_set.insert(3, "RunName", args.RunName)
    case_set.insert(4, "RunId", run_id)
    case_set.insert(5, "Split", args.Split)

    preferred_cols = [
        "RunTag",
        "FeatureConfig",
        "ModelName",
        "RunName",
        "RunId",
        "Split",
        "CaseGroup",
        "Order",
        "SelectionReason",
        "SampleId",
        "SplitOrder",
        "GroundTruthPixels",
        "PredictedPixels",
        "TruePositivePixels",
        "FalsePositivePixels",
        "FalseNegativePixels",
        "TotalErrorPixels",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "PredictionProbabilityMean",
        "PredictionProbabilityMax",
    ]

    cols = [c for c in preferred_cols if c in case_set.columns]
    other_cols = [c for c in case_set.columns if c not in cols]
    case_set = case_set[cols + other_cols]

    model_case_path = model_tables_dir / f"VisualizationCaseSet_{run_id}.csv"
    root_case_path = root_tables_dir / "VisualizationCaseSet.csv"

    case_set.to_csv(model_case_path, index=False)
    case_set.to_csv(root_case_path, index=False)

    print("")
    print("=== VISUALIZATION CASES SELECTED ===")
    print(f"RunTag: {args.RunTag}")
    print(f"FeatureConfig: {args.FeatureConfig}")
    print(f"RunId: {run_id}")
    print(f"Split: {args.Split}")
    print(f"Model cases: {model_case_path}")
    print(f"Root cases: {root_case_path}")
    print("")
    print(
        case_set[
            [
                "CaseGroup",
                "Order",
                "SampleId",
                "SelectionReason",
                "GroundTruthPixels",
                "PredictedPixels",
                "Dice",
                "IoU",
                "Precision",
                "Recall",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
