#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step19AuditProblemCases.py

Auditoría de casos problemáticos para MethaneProjectTFM.

Objetivos:
1) Verificar si un panel "Target" negro es un problema del dato raw, de la composición RGB
   o solo de la visualización.
2) Contar cuántas muestras tienen Target/Reference raw ausente, inválido o visualmente negro.
3) Contar anomalías de predicción por modelo/configuración/split:
   - predicción saturada: PredictedPositiveFraction >= umbral
   - predicción vacía/casi vacía si existen columnas
   - AreaRatio extremo
4) Generar un reporte detallado para un SampleId específico.

No reentrena ni re-predice modelos. Lee Features, Masks, SplitIndex, raw dataset y CSVs de métricas.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENTS = [
    {"ModelName": "SimpleUNet", "RunTag": "101622"},
    {"ModelName": "EnhancedUNet", "RunTag": "101622"},
    {"ModelName": "TransformerUNet", "RunTag": "101622"},
    {"ModelName": "TransformerPlus", "RunTag": "101840"},
]


def import_step13(project_root: Path):
    step13_path = project_root / "Scripts" / "Step13VisualizePredictions.py"
    if not step13_path.exists():
        raise FileNotFoundError(f"No existe Step13: {step13_path}")
    spec = importlib.util.spec_from_file_location("Step13VisualizePredictions", step13_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude importar Step13 desde {step13_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["Step13VisualizePredictions"] = module
    spec.loader.exec_module(module)
    return module


def ensure_sample_id_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SampleId" in out.columns:
        out["SampleId"] = out["SampleId"].astype(str)
        return out
    for candidate in ["sample_id", "id", "Id", "ID", "sampleId", "SampleID"]:
        if candidate in out.columns:
            out["SampleId"] = out[candidate].astype(str)
            return out
    raise KeyError(f"No encontré SampleId. Columnas: {list(out.columns)}")


def model_dir_matches(dir_name: str, requested_model_name: str) -> bool:
    if requested_model_name == "TransformerPlus":
        return dir_name.startswith("TransformerPlus_") or (
            dir_name.startswith("TransformerUNet_") and "TransformerPlus" in dir_name
        )
    if requested_model_name == "TransformerUNet":
        return dir_name.startswith("TransformerUNet_") and "TransformerPlus" not in dir_name
    return dir_name.startswith(f"{requested_model_name}_")


def find_model_roots(project_root: Path, run_tags: list[str], feature_configs: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tag_set = set(run_tags)
    for exp in DEFAULT_EXPERIMENTS:
        model_name = exp["ModelName"]
        run_tag = exp["RunTag"]
        if run_tag not in tag_set:
            continue
        for cfg in feature_configs:
            config_root = project_root / "Outputs" / "Experiments" / run_tag / cfg
            if not config_root.exists():
                continue
            candidates = []
            for p in config_root.iterdir():
                if p.is_dir() and model_dir_matches(p.name, model_name):
                    score = 0
                    if (p / "Checkpoints" / "BestModel.pt").exists():
                        score += 2
                    if (p / "Metrics").exists():
                        score += 1
                    if model_name == "TransformerPlus" and "TransformerPlus" in p.name:
                        score += 10
                    candidates.append((score, p.stat().st_mtime, p))
            if not candidates:
                continue
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            root = candidates[0][2]
            out.append({
                "RunTag": run_tag,
                "FeatureConfig": cfg,
                "ModelName": model_name,
                "RunId": root.name,
                "ModelRoot": root,
            })
    return out


def finite_stats(arr: Any) -> dict[str, Any]:
    if arr is None:
        return {
            "Available": False, "Shape": "", "FiniteFraction": np.nan,
            "Min": np.nan, "Max": np.nan, "Mean": np.nan, "Std": np.nan,
            "P02": np.nan, "P98": np.nan, "AllZero": np.nan
        }
    a = np.asarray(arr)
    shape = "x".join(map(str, a.shape))
    if a.size == 0:
        return {
            "Available": True, "Shape": shape, "FiniteFraction": 0.0,
            "Min": np.nan, "Max": np.nan, "Mean": np.nan, "Std": np.nan,
            "P02": np.nan, "P98": np.nan, "AllZero": np.nan
        }
    finite = np.isfinite(a)
    if not finite.any():
        return {
            "Available": True, "Shape": shape, "FiniteFraction": 0.0,
            "Min": np.nan, "Max": np.nan, "Mean": np.nan, "Std": np.nan,
            "P02": np.nan, "P98": np.nan, "AllZero": np.nan
        }
    v = a[finite].astype(float)
    return {
        "Available": True,
        "Shape": shape,
        "FiniteFraction": float(finite.mean()),
        "Min": float(np.nanmin(v)),
        "Max": float(np.nanmax(v)),
        "Mean": float(np.nanmean(v)),
        "Std": float(np.nanstd(v)),
        "P02": float(np.nanpercentile(v, 2)),
        "P98": float(np.nanpercentile(v, 98)),
        "AllZero": bool(np.nanmax(np.abs(v)) == 0),
    }


def get_metadata_from_raw(raw_sample: Any) -> dict[str, str]:
    result = {"Country": "", "Date": "", "Location": ""}
    if raw_sample is None:
        return result

    # Try dictionary-like objects recursively but safely.
    def get_any(obj, keys):
        if obj is None:
            return None
        if isinstance(obj, dict):
            for k in keys:
                if k in obj and obj[k] not in [None, ""]:
                    return obj[k]
            for key in ["metadata", "properties", "attrs"]:
                if key in obj:
                    val = get_any(obj[key], keys)
                    if val not in [None, ""]:
                        return val
        for k in keys:
            if hasattr(obj, k):
                val = getattr(obj, k)
                if val not in [None, ""]:
                    return val
        return None

    result["Country"] = str(get_any(raw_sample, ["country", "Country", "iso3_country", "admin_country"]) or "")
    result["Date"] = str(get_any(raw_sample, ["date", "Date", "datetime", "acquisition_date", "timestamp"]) or "")
    result["Location"] = str(get_any(raw_sample, ["location", "Location", "site", "source_name", "plume_id"]) or "")
    return result


def compute_rgb_context_stats(step13, raw_sample: Any, x_i: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    target_cube = step13.get_sample_array(raw_sample, "Target") if raw_sample is not None else None
    reference_cube = step13.get_sample_array(raw_sample, "Reference") if raw_sample is not None else None

    # Prefer Target; if absent, Reference.
    used = "Target" if target_cube is not None else ("Reference" if reference_cube is not None else "FeatureFallback")
    cube = target_cube if target_cube is not None else reference_cube

    stats_target = finite_stats(target_cube)
    stats_reference = finite_stats(reference_cube)

    rgb = None
    title = ""
    try:
        rgb, title = step13.make_swir_nir_blue_composite(cube, x_i, feature_names)
    except Exception as exc:
        title = f"RGB_ERROR:{type(exc).__name__}"

    rgb_stats = finite_stats(rgb)
    return {
        "ContextUsed": used,
        "RGBTitle": title,
        "TargetAvailable": stats_target["Available"],
        "TargetShape": stats_target["Shape"],
        "TargetFiniteFraction": stats_target["FiniteFraction"],
        "TargetMin": stats_target["Min"],
        "TargetMax": stats_target["Max"],
        "TargetMean": stats_target["Mean"],
        "TargetStd": stats_target["Std"],
        "TargetAllZero": stats_target["AllZero"],
        "ReferenceAvailable": stats_reference["Available"],
        "ReferenceShape": stats_reference["Shape"],
        "ReferenceFiniteFraction": stats_reference["FiniteFraction"],
        "ReferenceMin": stats_reference["Min"],
        "ReferenceMax": stats_reference["Max"],
        "ReferenceMean": stats_reference["Mean"],
        "ReferenceStd": stats_reference["Std"],
        "ReferenceAllZero": stats_reference["AllZero"],
        "RGBAvailable": rgb_stats["Available"],
        "RGBShape": rgb_stats["Shape"],
        "RGBFiniteFraction": rgb_stats["FiniteFraction"],
        "RGBMin": rgb_stats["Min"],
        "RGBMax": rgb_stats["Max"],
        "RGBMean": rgb_stats["Mean"],
        "RGBStd": rgb_stats["Std"],
        "RGBAllZero": rgb_stats["AllZero"],
    }


def selected_channel_stats(x_i: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    wanted = [
        "B8A", "B11", "B12", "NDSWIR", "RatioB12B11", "RatioB12B8A",
        "MBMP", "MBMPPlus", "DualEnhancementB12B11",
        "WindSpeed10m", "WindDirCos10m", "WindDirSin10m",
    ]

    out["FeatureChannels"] = int(x_i.shape[0])
    out["FeatureFiniteFraction"] = float(np.isfinite(x_i).mean())
    out["FeatureGlobalMin"] = float(np.nanmin(x_i))
    out["FeatureGlobalMax"] = float(np.nanmax(x_i))
    out["FeatureGlobalMean"] = float(np.nanmean(x_i))
    out["FeatureGlobalStd"] = float(np.nanstd(x_i))
    out["FeatureAllZero"] = bool(np.nanmax(np.abs(x_i[np.isfinite(x_i)])) == 0) if np.isfinite(x_i).any() else np.nan

    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    for name in wanted:
        if name not in name_to_idx:
            continue
        band = np.asarray(x_i[name_to_idx[name]], dtype=float)
        out[f"{name}_Mean"] = float(np.nanmean(band))
        out[f"{name}_Std"] = float(np.nanstd(band))
        out[f"{name}_Min"] = float(np.nanmin(band))
        out[f"{name}_Max"] = float(np.nanmax(band))
        out[f"{name}_AllZero"] = bool(np.nanmax(np.abs(band[np.isfinite(band)])) == 0) if np.isfinite(band).any() else np.nan
    return out


def audit_feature_context(
    *,
    project_root: Path,
    output_dir: Path,
    step13,
    splits: list[str],
    feature_configs: list[str],
    run_tags: list[str],
    sample_id: str | None,
    no_raw_context: bool,
    black_mean: float,
    black_std: float,
    max_samples: int | None,
) -> pd.DataFrame:
    # Use first available run_tag for each configuration to avoid duplicate feature scans.
    rows: list[dict[str, Any]] = []

    dataset = None
    sample_table = None
    if not no_raw_context and getattr(step13, "HAS_TACO_READERS", False):
        try:
            project_config = step13.load_yaml(project_root / "Configs" / "ProjectConfig.yaml")
            data_root, dataset_name = step13.resolve_dataset_location(project_config)
            dataset, sample_table = step13.load_dataset_flexible(data_root, dataset_name)
            print(f"[INFO] Raw dataset cargado: {dataset_name}")
        except Exception as exc:
            print(f"[WARN] No pude cargar raw dataset: {repr(exc)}")

    for run_tag in run_tags:
        run_root = project_root / "Outputs" / "Experiments" / run_tag
        if not run_root.exists():
            continue

        for cfg in feature_configs:
            feature_dir = run_root / cfg / "Features"
            if not feature_dir.exists():
                continue

            try:
                feature_names = step13.load_feature_names(cfg)
            except Exception:
                feature_names = []

            for split in splits:
                x_path = feature_dir / f"{split}Features.npy"
                y_path = feature_dir / f"{split}Masks.npy"
                if not x_path.exists() or not y_path.exists():
                    continue

                x = np.load(x_path, mmap_mode="r")
                y = np.load(y_path, mmap_mode="r")
                split_index = step13.load_split_index(run_root, split, expected_n=int(x.shape[0]))
                split_index = ensure_sample_id_column(split_index)

                if sample_id:
                    sub = split_index[split_index["SampleId"].astype(str).eq(str(sample_id))]
                else:
                    sub = split_index.copy()
                    if max_samples is not None and len(sub) > max_samples:
                        sub = sub.head(max_samples)

                print(f"[INFO] Auditando contexto: RunTag={run_tag} Config={cfg} Split={split} muestras={len(sub)}")

                for _, idx_row in sub.iterrows():
                    sid = str(idx_row["SampleId"])
                    order = int(idx_row["SplitOrder"]) if "SplitOrder" in idx_row else int(idx_row.name)
                    x_i = np.array(x[order], dtype=np.float32, copy=True)
                    y_i = np.array(y[order], dtype=np.uint8, copy=True)

                    raw_sample = None
                    meta = {"Country": "", "Date": "", "Location": ""}
                    if dataset is not None:
                        try:
                            raw_sample = step13.read_sample_flexible(dataset, sample_table, sid)
                            meta = get_metadata_from_raw(raw_sample)
                        except Exception as exc:
                            raw_sample = None

                    context_stats = compute_rgb_context_stats(step13, raw_sample, x_i, feature_names)
                    chan_stats = selected_channel_stats(x_i, feature_names)

                    gt_pixels = int((y_i > 0).sum())
                    rgb_mean = context_stats.get("RGBMean", np.nan)
                    rgb_std = context_stats.get("RGBStd", np.nan)
                    rgb_black = bool(
                        (not context_stats.get("RGBAvailable", False)) or
                        (np.isfinite(rgb_mean) and np.isfinite(rgb_std) and rgb_mean <= black_mean and rgb_std <= black_std)
                    )

                    target_suspect = bool(
                        (not context_stats.get("TargetAvailable", False)) or
                        context_stats.get("TargetAllZero", False) is True or
                        context_stats.get("TargetFiniteFraction", 1.0) < 0.95
                    )

                    rows.append({
                        "RunTag": run_tag,
                        "FeatureConfig": cfg,
                        "Split": split,
                        "SampleId": sid,
                        "SplitOrder": order,
                        **meta,
                        "GTPixels": gt_pixels,
                        "GTPositiveFraction": gt_pixels / float(y_i.size),
                        "RGBBlackOrInvalid": rgb_black,
                        "TargetSuspect": target_suspect,
                        **context_stats,
                        **chan_stats,
                    })

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "Step19SampleQualityAudit.csv"
    df.to_csv(out_path, index=False)
    print(f"[OUT] {out_path}")
    return df


def audit_metrics(
    *,
    project_root: Path,
    output_dir: Path,
    step13,
    splits: list[str],
    run_tags: list[str],
    feature_configs: list[str],
    saturated_threshold: float,
    area_ratio_threshold: float,
    sample_id: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roots = find_model_roots(project_root, run_tags, feature_configs)

    for item in roots:
        model_root: Path = item["ModelRoot"]
        run_tag = item["RunTag"]
        cfg = item["FeatureConfig"]
        model = item["ModelName"]
        run_root = project_root / "Outputs" / "Experiments" / run_tag

        for split in splits:
            path = model_root / "Metrics" / f"{split}MetricsBySample.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            try:
                split_index = step13.load_split_index(run_root, split, expected_n=len(df))
            except Exception:
                split_index = None

            if "SampleId" not in df.columns and split_index is not None and len(split_index) == len(df):
                df["SampleId"] = split_index["SampleId"].values
            df = ensure_sample_id_column(df)

            if sample_id:
                df = df[df["SampleId"].astype(str).eq(str(sample_id))].copy()

            for _, r in df.iterrows():
                pred_frac = np.nan
                if "PredictedPositiveFraction" in r.index:
                    pred_frac = pd.to_numeric(r["PredictedPositiveFraction"], errors="coerce")
                elif "PredictedPixels" in r.index:
                    pred_frac = pd.to_numeric(r["PredictedPixels"], errors="coerce") / 40000.0

                area_ratio = pd.to_numeric(r.get("AreaRatio", np.nan), errors="coerce")
                pred_pixels = pd.to_numeric(r.get("PredictedPixels", r.get("PredPixels", np.nan)), errors="coerce")
                gt_pixels = pd.to_numeric(r.get("GroundTruthPixels", r.get("GTPixels", r.get("GTArea", np.nan))), errors="coerce")
                dice = pd.to_numeric(r.get("Dice", np.nan), errors="coerce")
                iou = pd.to_numeric(r.get("IoU", np.nan), errors="coerce")
                precision = pd.to_numeric(r.get("Precision", np.nan), errors="coerce")
                recall = pd.to_numeric(r.get("Recall", np.nan), errors="coerce")

                rows.append({
                    "RunTag": run_tag,
                    "FeatureConfig": cfg,
                    "ModelName": model,
                    "RunId": item["RunId"],
                    "Split": split,
                    "SampleId": str(r["SampleId"]),
                    "Dice": dice,
                    "IoU": iou,
                    "Precision": precision,
                    "Recall": recall,
                    "GTPixels": gt_pixels,
                    "PredPixels": pred_pixels,
                    "AreaRatio": area_ratio,
                    "PredictedPositiveFraction": pred_frac,
                    "IsSaturatedPrediction": bool(np.isfinite(pred_frac) and pred_frac >= saturated_threshold),
                    "IsNearEmptyPrediction": bool(np.isfinite(pred_frac) and pred_frac <= 0.0005),
                    "IsExtremeAreaRatio": bool(np.isfinite(area_ratio) and area_ratio >= area_ratio_threshold),
                })

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "Step19MetricsAnomalyAudit.csv"
    out.to_csv(out_path, index=False)
    print(f"[OUT] {out_path}")
    return out


def write_problem_report(
    *,
    output_dir: Path,
    sample_id: str,
    context_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append(f"Problem sample audit: {sample_id}")
    lines.append("=" * 90)
    lines.append("")

    csub = context_df[context_df["SampleId"].astype(str).eq(str(sample_id))].copy()
    msub = metrics_df[metrics_df["SampleId"].astype(str).eq(str(sample_id))].copy()

    lines.append("Context / raw / feature audit")
    lines.append("-" * 90)
    if csub.empty:
        lines.append("No rows found in context audit.")
    else:
        cols = [
            "RunTag", "FeatureConfig", "Split", "Country", "Date", "Location",
            "GTPixels", "ContextUsed", "RGBTitle", "RGBBlackOrInvalid",
            "TargetSuspect", "TargetAvailable", "TargetShape", "TargetFiniteFraction",
            "TargetMin", "TargetMax", "TargetMean", "TargetStd", "TargetAllZero",
            "ReferenceAvailable", "ReferenceShape", "RGBMean", "RGBStd",
            "FeatureChannels", "FeatureFiniteFraction", "FeatureGlobalMin", "FeatureGlobalMax",
            "FeatureGlobalMean", "FeatureGlobalStd", "FeatureAllZero",
        ]
        cols = [c for c in cols if c in csub.columns]
        lines.append(csub[cols].to_string(index=False))

        channel_cols = [c for c in csub.columns if any(c.startswith(prefix) for prefix in [
            "B8A_", "B11_", "B12_", "NDSWIR_", "RatioB12B11_", "RatioB12B8A_",
            "MBMP_", "MBMPPlus_", "DualEnhancementB12B11_", "WindSpeed10m_",
            "WindDirCos10m_", "WindDirSin10m_"
        ])]
        if channel_cols:
            lines.append("")
            lines.append("Selected feature channel statistics")
            lines.append("-" * 90)
            lines.append(csub[["RunTag", "FeatureConfig", "Split"] + channel_cols].to_string(index=False))

    lines.append("")
    lines.append("Prediction metrics audit")
    lines.append("-" * 90)
    if msub.empty:
        lines.append("No rows found in metrics audit.")
    else:
        cols = [
            "RunTag", "FeatureConfig", "ModelName", "Split", "Dice", "IoU",
            "Precision", "Recall", "GTPixels", "PredPixels", "AreaRatio",
            "PredictedPositiveFraction", "IsSaturatedPrediction",
            "IsNearEmptyPrediction", "IsExtremeAreaRatio"
        ]
        cols = [c for c in cols if c in msub.columns]
        lines.append(msub[cols].sort_values(["Split", "FeatureConfig", "ModelName"]).to_string(index=False))

    report_path = output_dir / f"Step19ProblemSampleReport_{sample_id}.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OUT] {report_path}")


def print_summary(context_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    print("\n=== SUMMARY: context quality ===")
    if not context_df.empty:
        summary = (
            context_df.groupby(["Split", "RunTag", "FeatureConfig"], dropna=False)
            .agg(
                Samples=("SampleId", "nunique"),
                RGBBlackOrInvalid=("RGBBlackOrInvalid", "sum"),
                TargetSuspect=("TargetSuspect", "sum"),
                TargetUnavailable=("TargetAvailable", lambda s: int((~s.astype(bool)).sum())),
                FeatureAllZero=("FeatureAllZero", "sum"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))

        bad = context_df[context_df["RGBBlackOrInvalid"].astype(bool) | context_df["TargetSuspect"].astype(bool)]
        if len(bad):
            print("\nProblematic context rows:")
            cols = ["Split", "RunTag", "FeatureConfig", "SampleId", "Country", "Date", "Location", "RGBBlackOrInvalid", "TargetSuspect", "ContextUsed", "RGBMean", "RGBStd", "TargetMean", "TargetStd", "FeatureGlobalStd"]
            cols = [c for c in cols if c in bad.columns]
            print(bad[cols].head(50).to_string(index=False))
            if len(bad) > 50:
                print(f"... {len(bad)-50} more rows in CSV")

    print("\n=== SUMMARY: prediction anomalies ===")
    if not metrics_df.empty:
        summary = (
            metrics_df.groupby(["Split", "FeatureConfig", "ModelName"], dropna=False)
            .agg(
                Samples=("SampleId", "nunique"),
                Saturated=("IsSaturatedPrediction", "sum"),
                NearEmpty=("IsNearEmptyPrediction", "sum"),
                ExtremeAreaRatio=("IsExtremeAreaRatio", "sum"),
                MeanDice=("Dice", "mean"),
                MeanPredFrac=("PredictedPositiveFraction", "mean"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))

        sat = metrics_df[metrics_df["IsSaturatedPrediction"].astype(bool)]
        if len(sat):
            print("\nSaturated prediction rows:")
            cols = ["Split", "FeatureConfig", "ModelName", "SampleId", "Dice", "Precision", "Recall", "GTPixels", "PredPixels", "PredictedPositiveFraction", "AreaRatio"]
            print(sat[cols].head(80).to_string(index=False))
            if len(sat) > 80:
                print(f"... {len(sat)-80} more rows in CSV")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita casos problemáticos de datos, contexto visual y métricas.")
    parser.add_argument("--ProjectRoot", default=".")
    parser.add_argument("--OutputDir", default="Outputs/ResultsChapter_101622_101840")
    parser.add_argument("--RunTags", default="101622,101840")
    parser.add_argument("--FeatureConfigs", default="ConfigB,ConfigC")
    parser.add_argument("--Splits", default="Test")
    parser.add_argument("--SampleId", default=None)
    parser.add_argument("--NoRawContext", action="store_true")
    parser.add_argument("--BlackMeanThreshold", type=float, default=0.02)
    parser.add_argument("--BlackStdThreshold", type=float, default=0.01)
    parser.add_argument("--SaturatedThreshold", type=float, default=0.99)
    parser.add_argument("--AreaRatioThreshold", type=float, default=20.0)
    parser.add_argument("--MaxSamples", type=int, default=None, help="Opcional para prueba rápida; por defecto audita todo el split.")
    args = parser.parse_args()

    project_root = Path(args.ProjectRoot).resolve()
    output_dir = Path(args.OutputDir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    tables_dir = output_dir / "Tables"
    logs_dir = output_dir / "Logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    step13 = import_step13(project_root)

    run_tags = [x.strip() for x in args.RunTags.split(",") if x.strip()]
    feature_configs = [x.strip() for x in args.FeatureConfigs.split(",") if x.strip()]
    splits = [x.strip() for x in args.Splits.split(",") if x.strip()]

    print("\n=== STEP19 AUDIT START ===")
    print(f"ProjectRoot: {project_root}")
    print(f"OutputDir: {output_dir}")
    print(f"Splits: {splits}")
    print(f"FeatureConfigs: {feature_configs}")
    print(f"SampleId: {args.SampleId or 'ALL'}")

    context_df = audit_feature_context(
        project_root=project_root,
        output_dir=tables_dir,
        step13=step13,
        splits=splits,
        feature_configs=feature_configs,
        run_tags=run_tags,
        sample_id=args.SampleId,
        no_raw_context=args.NoRawContext,
        black_mean=args.BlackMeanThreshold,
        black_std=args.BlackStdThreshold,
        max_samples=args.MaxSamples,
    )

    metrics_df = audit_metrics(
        project_root=project_root,
        output_dir=tables_dir,
        step13=step13,
        splits=splits,
        run_tags=run_tags,
        feature_configs=feature_configs,
        saturated_threshold=args.SaturatedThreshold,
        area_ratio_threshold=args.AreaRatioThreshold,
        sample_id=args.SampleId,
    )

    print_summary(context_df, metrics_df)

    if args.SampleId:
        write_problem_report(
            output_dir=tables_dir,
            sample_id=args.SampleId,
            context_df=context_df,
            metrics_df=metrics_df,
        )

    run_config = {
        "ProjectRoot": str(project_root),
        "OutputDir": str(output_dir),
        "RunTags": run_tags,
        "FeatureConfigs": feature_configs,
        "Splits": splits,
        "SampleId": args.SampleId,
        "BlackMeanThreshold": args.BlackMeanThreshold,
        "BlackStdThreshold": args.BlackStdThreshold,
        "SaturatedThreshold": args.SaturatedThreshold,
        "AreaRatioThreshold": args.AreaRatioThreshold,
        "MaxSamples": args.MaxSamples,
    }
    (logs_dir / "Step19AuditProblemCasesRunConfig.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== STEP19 AUDIT COMPLETED ===")
    print(f"ContextAuditCSV: {tables_dir / 'Step19SampleQualityAudit.csv'}")
    print(f"MetricsAuditCSV: {tables_dir / 'Step19MetricsAnomalyAudit.csv'}")
    if args.SampleId:
        print(f"ProblemReport: {tables_dir / f'Step19ProblemSampleReport_{args.SampleId}.txt'}")


if __name__ == "__main__":
    main()
