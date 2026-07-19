#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step17AnalyzeResultsForChapter.py

Construye resultados, tablas LaTeX y figuras para el capítulo de Resultados del
TFM, organizando el análisis experimento por experimento.

Experimentos incluidos:
    E1: U-Net simple       Configuración B vs Configuración C
    E2: U-Net mejorada     Configuración B vs Configuración C
    E3: U-Net Transformer  Configuración B vs Configuración C
    E4: TransformerPlus    Configuración B vs Configuración C

RunTags esperados:
    101622, 101840, Exp101840

Diseño visual:
    - Español en títulos, ejes y leyendas.
    - Etiquetas cortas.
    - Paleta azul.
    - Fuente Montserrat si está disponible.
    - Figuras limpias para memoria TFM.
    - Tablas técnicas en LaTeX, no como imagen.

Modos:
    inventory   Inventario de experimentos y evidencia disponible.
    master      Construye tablas maestras si hay métricas por muestra.
    experiment  Genera análisis de un experimento o todos.
    factors     Genera análisis por fluxrate, tamaño de pluma y viento.
    summary     Genera síntesis global.
    all         Ejecuta inventory + master + experiment + factors + summary.

Ejemplos:
    python Scripts/Step17AnalyzeResultsForChapter.py --mode inventory

    python Scripts/Step17AnalyzeResultsForChapter.py --mode experiment --Experiment SimpleUNet

    python Scripts/Step17AnalyzeResultsForChapter.py --mode all \
        --ProjectRoot /data/users/kabasmen/MethaneProjectTFM \
        --OutputDir Outputs/ResultsChapter_101622_101840 \
        --RunTags 101622,101840,Exp101840
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


# =============================================================================
# Configuración visual
# =============================================================================

FONT_FAMILY = "Montserrat"

PALETTE = {
    "navy": "#0B2A63",
    "blue": "#1F65C8",
    "blue2": "#2C7DE1",
    "cyan": "#5BB7E8",
    "cyan2": "#8BD9E8",
    "pale": "#EDF5FF",
    "grid": "#E5EEF8",
    "text": "#16325C",
    "muted": "#4F6787",
    "white": "#FFFFFF",
}

MODEL_COLOR = {
    "SimpleUNet": "#8BD9E8",
    "EnhancedUNet": "#5BB7E8",
    "TransformerUNet": "#2C7DE1",
    "TransformerPlus": "#0B2A63",
}

CONFIG_COLOR = {
    "ConfigB": "#1F65C8",
    "ConfigC": "#5BB7E8",
}

SPANISH_MODEL = {
    "SimpleUNet": "U-Net simple",
    "EnhancedUNet": "U-Net mejorada",
    "TransformerUNet": "U-Net Transformer",
    "TransformerPlus": "TransformerPlus",
}

SHORT_MODEL = {
    "SimpleUNet": "U-Net simple",
    "EnhancedUNet": "U-Net mejorada",
    "TransformerUNet": "U-Net Transformer",
    "TransformerPlus": "TransformerPlus",
}

SPANISH_CONFIG = {
    "ConfigB": "Configuración B",
    "ConfigC": "Configuración C",
}

SHORT_CONFIG = {
    "ConfigB": "B",
    "ConfigC": "C",
}

EXPERIMENTS = {
    "SimpleUNet": {
        "ExperimentId": "E1",
        "Order": 1,
        "Slug": "Experimento01UnetSimple",
        "Title": "Experimento 1: U-Net simple",
        "SpanishModel": "U-Net simple",
    },
    "EnhancedUNet": {
        "ExperimentId": "E2",
        "Order": 2,
        "Slug": "Experimento02UnetMejorada",
        "Title": "Experimento 2: U-Net mejorada",
        "SpanishModel": "U-Net mejorada",
    },
    "TransformerUNet": {
        "ExperimentId": "E3",
        "Order": 3,
        "Slug": "Experimento03UnetTransformer",
        "Title": "Experimento 3: U-Net Transformer",
        "SpanishModel": "U-Net Transformer",
    },
    "TransformerPlus": {
        "ExperimentId": "E4",
        "Order": 4,
        "Slug": "Experimento04TransformerPlus",
        "Title": "Experimento 4: TransformerPlus",
        "SpanishModel": "TransformerPlus",
    },
}

EPS = 1e-8

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

QUALITY_RULES = {
    "acceptable_dice": 0.30,
    "acceptable_area_low": 0.50,
    "acceptable_area_high": 2.00,
    "near_empty_ratio": 0.10,
    "severe_fn_recall": 0.20,
    "severe_fp_precision": 0.30,
    "oversegmentation_ratio": 3.00,
    "undersegmentation_ratio": 0.33,
    "saturation_fraction": 0.25,
}

WIND_ALIGNMENT_BINS = [0.0, 0.33, 0.66, 1.0]
WIND_ALIGNMENT_LABELS = ["Baja", "Media", "Alta"]

QUANTILE_BIN_LABELS_3 = ["Bajo", "Medio", "Alto"]
PLUME_SIZE_LABELS_3 = ["Pequeña", "Media", "Grande"]
UNCERTAINTY_LABELS_3 = ["Baja", "Media", "Alta"]


def setup_style() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = FONT_FAMILY if FONT_FAMILY in available else "DejaVu Sans"

    plt.rcParams.update({
        "font.family": font,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": PALETTE["text"],
        "axes.linewidth": 0.8,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, alpha=0.95)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Espacio adicional para títulos en dos líneas y leyendas inferiores.
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.20)
    plt.close(fig)
    print(f"[FIG] {path}")


# =============================================================================
# Utilidades
# =============================================================================

def pascal_case(text: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", str(text))
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "Base": output_dir,
        "Tables": output_dir / "Tables",
        "Figures": output_dir / "Figures",
        "Latex": output_dir / "Latex",
        "Logs": output_dir / "Logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] No se pudo leer {path}: {exc}")
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path, digits: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if digits is not None:
        for col in out.select_dtypes(include=[np.number]).columns:
            out[col] = out[col].round(digits)
    out.to_csv(path, index=False)
    print(f"[CSV] {path}")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[TXT] {path}")


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    mapping = {normalize(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize(candidate)
        if key in mapping:
            return mapping[key]
    for candidate in candidates:
        key = normalize(candidate)
        for normed, original in mapping.items():
            if key and key in normed:
                return original
    return None


def infer_model_name(value: str) -> str:
    v = str(value).lower()
    if "transformerplus" in v or "transformer_plus" in v or "plus" in v:
        return "TransformerPlus"
    if "transformerunet" in v or "transformer" in v:
        return "TransformerUNet"
    if "enhanced" in v or "mejorada" in v:
        return "EnhancedUNet"
    if "simple" in v:
        return "SimpleUNet"
    return str(value)


def infer_run_tag_from_path(path: Path, run_tags: Sequence[str]) -> Optional[str]:
    s = str(path)
    for tag in run_tags:
        if tag and tag in s:
            return tag
    return None


def infer_config_from_path(path: Path) -> Optional[str]:
    s = str(path)
    if "ConfigB" in s:
        return "ConfigB"
    if "ConfigC" in s:
        return "ConfigC"
    return None


def format_float(value, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def short_label(model_name: str, config: str = "") -> str:
    m = SHORT_MODEL.get(str(model_name), str(model_name))
    if config:
        return f"{SHORT_CONFIG.get(str(config), str(config))} · {m}"
    return m


def spanish_config(config: str) -> str:
    return SPANISH_CONFIG.get(str(config), str(config))


def spanish_model(model: str) -> str:
    return SPANISH_MODEL.get(str(model), str(model))


# =============================================================================
# Estandarización de tablas
# =============================================================================

def standardize_columns(df: pd.DataFrame, source_path: Optional[Path] = None, run_tags: Sequence[str] = ()) -> pd.DataFrame:
    out = df.copy()

    aliases = {
        "RunTag": ["RunTag", "run_tag", "ExperimentTag", "ExpTag"],
        "FeatureConfig": ["FeatureConfig", "Config", "InputConfig", "feature_config"],
        "ModelName": ["ModelName", "Model", "model_name", "Architecture"],
        "ModelRunId": ["ModelRunId", "RunId", "model_run_id", "ExperimentId"],
        "RunName": ["RunName", "run_name"],
        "Threshold": ["Threshold", "threshold", "EvalThreshold"],
        "SampleId": ["SampleId", "SampleID", "sample_id", "sample", "ItemId", "TacoId"],

        "BestThreshold": ["BestThreshold", "Threshold", "threshold", "EvalThreshold"],
        "Samples": ["Samples", "SampleCount", "NumSamples", "N"],

        "MeanDice": ["MeanDice", "TestMeanDice", "DiceMean", "mean_dice"],
        "MeanIoU": ["MeanIoU", "TestMeanIoU", "IoUMean", "mean_iou"],
        "MeanPrecision": ["MeanPrecision", "PrecisionMean", "mean_precision"],
        "MeanRecall": ["MeanRecall", "RecallMean", "mean_recall"],
        "GlobalDice": ["GlobalDice", "TestGlobalDice", "global_dice"],
        "GlobalIoU": ["GlobalIoU", "TestGlobalIoU", "global_iou"],

        "Dice": ["Dice", "SampleDice", "dice"],
        "IoU": ["IoU", "SampleIoU", "iou", "Jaccard"],
        "Precision": ["Precision", "precision"],
        "Recall": ["Recall", "recall"],

        "TP": ["TP", "TruePositive", "TruePositives", "TruePositivePixels", "tp"],
        "FP": ["FP", "FalsePositive", "FalsePositives", "FalsePositivePixels", "fp"],
        "FN": ["FN", "FalseNegative", "FalseNegatives", "FalseNegativePixels", "fn"],
        "TN": ["TN", "TrueNegative", "TrueNegatives", "TrueNegativePixels", "tn"],

        "GTArea": ["GTArea", "GtArea", "GroundTruthArea", "GroundTruthPixels", "MaskArea", "PlumeArea", "TargetArea"],
        "PredArea": ["PredArea", "PredictedArea", "PredictedPixels", "PredictionArea"],

        "PredictedPositiveFraction": ["PredictedPositiveFraction", "PositiveFraction", "PredPositiveFraction", "PredictedPosFrac"],
        "ProbabilityMean": ["ProbabilityMean", "ProbMean", "MeanProbability"],
        "ProbabilityMax": ["ProbabilityMax", "ProbMax", "MaxProbability"],
        "ProbabilityP95": ["ProbabilityP95", "ProbP95", "P95Probability"],

        "Epoch": ["Epoch", "epoch"],
        "TrainLoss": ["TrainLoss", "train_loss"],
        "ValidationLoss": ["ValidationLoss", "ValLoss", "validation_loss", "val_loss"],
        "TrainMeanDice": ["TrainMeanDice", "TrainDice"],
        "ValidationMeanDice": ["ValidationMeanDice", "ValDice", "ValidationDice"],
        "TrainMeanIoU": ["TrainMeanIoU", "TrainIoU"],
        "ValidationMeanIoU": ["ValidationMeanIoU", "ValIoU", "ValidationIoU"],

        "Fluxrate": ["Fluxrate", "FluxRate", "flux_rate", "emission_rate", "EmissionRate", "source_rate", "SourceRate", "kg_h", "kg_per_h", "ch4_flux", "ch4_fluxrate", "l0:detection:ch4_fluxrate"],
        "FluxrateStd": ["FluxrateStd", "FluxrateSTD", "FluxrateStandardDeviation", "ch4_fluxrate_std", "l0:detection:ch4_fluxrate_std"],
        "Country": ["Country", "country", "l0:location:country"],
        "Site": ["Site", "site", "l0:location:site"],
        "WindSpeed10m": ["WindSpeed10m", "WindSpeed", "wind_speed", "wind_speed_10m"],
        "WindAlignment": ["WindAlignment", "PlumeWindAlignment", "AxialAlignment", "Alignment"],
        "WindSource": ["WindSource", "wind_source", "DetectionWindSource"],
    }

    for std, candidates in aliases.items():
        if std not in out.columns:
            col = find_column(out, candidates)
            if col is not None:
                out[std] = out[col]

    if "RunTag" not in out.columns and source_path is not None:
        tag = infer_run_tag_from_path(source_path, run_tags)
        if tag:
            out["RunTag"] = tag

    if "FeatureConfig" not in out.columns and source_path is not None:
        config = infer_config_from_path(source_path)
        if config:
            out["FeatureConfig"] = config

    if "ModelName" not in out.columns:
        if "ModelRunId" in out.columns:
            out["ModelName"] = out["ModelRunId"].apply(infer_model_name)
        elif source_path is not None:
            out["ModelName"] = infer_model_name(str(source_path))

    if "SampleId" in out.columns:
        out["SampleId"] = out["SampleId"].astype(str)

    numeric_cols = [
        "BestThreshold", "Threshold", "Samples",
        "MeanDice", "MeanIoU", "MeanPrecision", "MeanRecall", "GlobalDice", "GlobalIoU",
        "Dice", "IoU", "Precision", "Recall",
        "TP", "FP", "FN", "TN", "GTArea", "PredArea",
        "PredictedPositiveFraction", "ProbabilityMean", "ProbabilityMax", "ProbabilityP95",
        "Epoch", "TrainLoss", "ValidationLoss", "TrainMeanDice", "ValidationMeanDice",
        "TrainMeanIoU", "ValidationMeanIoU",
        "Fluxrate", "FluxrateStd", "WindSpeed10m", "WindAlignment",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


# =============================================================================
# Descubrimiento de experimentos
# =============================================================================

@dataclass
class RunRecord:
    RunTag: str
    FeatureConfig: str
    ModelName: str
    ModelRunId: str
    ModelRoot: Path
    MetricsDirectory: Path
    TablesDirectory: Path


def discover_runs(project_root: Path, run_tags: Sequence[str]) -> pd.DataFrame:
    rows = []
    exp_root = project_root / "Outputs" / "Experiments"

    for tag in run_tags:
        tag_dir = exp_root / tag
        if not tag_dir.exists():
            continue

        for test_summary in tag_dir.rglob("Metrics/TestMetricsSummary.csv"):
            model_root = test_summary.parents[1]
            config = None
            for part in model_root.parts:
                if part in {"ConfigB", "ConfigC"}:
                    config = part
            if config not in {"ConfigB", "ConfigC"}:
                continue

            model_run_id = model_root.name
            model_name = infer_model_name(model_run_id)

            if model_name not in EXPERIMENTS:
                continue

            rows.append({
                "RunTag": tag,
                "FeatureConfig": config,
                "ModelName": model_name,
                "SpanishModel": spanish_model(model_name),
                "ShortLabel": short_label(model_name, config),
                "ExperimentId": EXPERIMENTS[model_name]["ExperimentId"],
                "ExperimentOrder": EXPERIMENTS[model_name]["Order"],
                "ExperimentTitle": EXPERIMENTS[model_name]["Title"],
                "ModelRunId": model_run_id,
                "ModelRoot": str(model_root),
                "MetricsDirectory": str(model_root / "Metrics"),
                "TablesDirectory": str(model_root / "Tables"),
                "HasTrainingHistory": (model_root / "Metrics" / "TrainingHistory.csv").exists(),
                "HasTestSummary": (model_root / "Metrics" / "TestMetricsSummary.csv").exists(),
                "HasTestBySample": (model_root / "Metrics" / "TestMetricsBySample.csv").exists(),
                "HasPredictionFigureIndex": (model_root / "Tables" / "PredictionFigureIndex.csv").exists(),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ExperimentOrder", "FeatureConfig", "RunTag"]).reset_index(drop=True)
    return df


def load_run_tables(row: pd.Series, run_tags: Sequence[str]) -> Dict[str, pd.DataFrame]:
    metrics_dir = Path(row["MetricsDirectory"])
    tables_dir = Path(row["TablesDirectory"])

    files = {
        "TrainingHistory": metrics_dir / "TrainingHistory.csv",
        "BestEpochSummary": metrics_dir / "BestEpochSummary.csv",
        "TestMetricsSummary": metrics_dir / "TestMetricsSummary.csv",
        "TestMetricsBySample": metrics_dir / "TestMetricsBySample.csv",
        "PredictionFigureIndex": tables_dir / "PredictionFigureIndex.csv",
        "ModelRunSummary": tables_dir / "ModelRunSummary.csv",
    }

    out = {}
    for name, path in files.items():
        if path.exists():
            out[name] = standardize_columns(read_csv(path), path, run_tags)
        else:
            out[name] = pd.DataFrame()
    return out


# =============================================================================
# Métricas derivadas
# =============================================================================

def add_binary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if all(c in out.columns for c in ["TP", "FP", "FN"]):
        if "Dice" not in out.columns or out["Dice"].isna().all():
            out["Dice"] = (2 * out["TP"]) / (2 * out["TP"] + out["FP"] + out["FN"] + EPS)
        if "IoU" not in out.columns or out["IoU"].isna().all():
            out["IoU"] = out["TP"] / (out["TP"] + out["FP"] + out["FN"] + EPS)
        if "Precision" not in out.columns or out["Precision"].isna().all():
            out["Precision"] = out["TP"] / (out["TP"] + out["FP"] + EPS)
        if "Recall" not in out.columns or out["Recall"].isna().all():
            out["Recall"] = out["TP"] / (out["TP"] + out["FN"] + EPS)

    if all(c in out.columns for c in ["TP", "FP", "FN", "TN"]):
        out["Specificity"] = out["TN"] / (out["TN"] + out["FP"] + EPS)
        out["FalsePositiveRate"] = out["FP"] / (out["FP"] + out["TN"] + EPS)
        out["FalseNegativeRate"] = out["FN"] / (out["FN"] + out["TP"] + EPS)
        out["BalancedAccuracy"] = 0.5 * (out["Recall"] + out["Specificity"])
        denom = np.sqrt((out["TP"] + out["FP"]) * (out["TP"] + out["FN"]) * (out["TN"] + out["FP"]) * (out["TN"] + out["FN"]) + EPS)
        out["MCC"] = ((out["TP"] * out["TN"]) - (out["FP"] * out["FN"])) / denom

    if "GTArea" not in out.columns and all(c in out.columns for c in ["TP", "FN"]):
        out["GTArea"] = out["TP"] + out["FN"]

    if "PredArea" not in out.columns and all(c in out.columns for c in ["TP", "FP"]):
        out["PredArea"] = out["TP"] + out["FP"]

    if all(c in out.columns for c in ["GTArea", "PredArea"]):
        out["AreaRatio"] = out["PredArea"] / (out["GTArea"] + EPS)
        out["AbsoluteAreaError"] = (out["PredArea"] - out["GTArea"]).abs()
        out["RelativeAreaError"] = (out["PredArea"] - out["GTArea"]) / (out["GTArea"] + EPS)

    if all(c in out.columns for c in ["FP", "GTArea"]):
        out["FPBurden"] = out["FP"] / (out["GTArea"] + EPS)

    if all(c in out.columns for c in ["FN", "GTArea"]):
        out["FNBurden"] = out["FN"] / (out["GTArea"] + EPS)

    if "PredictedPositiveFraction" not in out.columns and "PredArea" in out.columns:
        out["PredictedPositiveFraction"] = out["PredArea"] / 40000.0

    return out


def safe_qcut(series: pd.Series, q: int, labels: Sequence[str]) -> pd.Series:
    """Quantile binning robust to repeated values and small samples."""
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = s.dropna()
    if len(valid) < max(10, q * 3):
        return out
    try:
        binned = pd.qcut(valid, q=q, labels=labels, duplicates="drop")
        out.loc[valid.index] = binned.astype(str)
    except Exception:
        try:
            binned = pd.cut(valid, bins=q, labels=labels)
            out.loc[valid.index] = binned.astype(str)
        except Exception:
            pass
    return out


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "GTArea" in out.columns:
        out["PlumeSizeBin"] = safe_qcut(out["GTArea"], 3, PLUME_SIZE_LABELS_3)

    if "Fluxrate" in out.columns:
        out["FluxrateBin"] = safe_qcut(out["Fluxrate"], 3, QUANTILE_BIN_LABELS_3)

    if "FluxrateRelativeUncertainty" in out.columns:
        out["FluxrateUncertaintyBin"] = safe_qcut(out["FluxrateRelativeUncertainty"], 3, UNCERTAINTY_LABELS_3)

    if "WindSpeed10m" in out.columns:
        out["WindSpeedBin"] = safe_qcut(out["WindSpeed10m"], 3, QUANTILE_BIN_LABELS_3)

    if "WindAlignment" in out.columns:
        vals = pd.to_numeric(out["WindAlignment"], errors="coerce").clip(lower=0, upper=1)
        out["WindAlignmentBin"] = pd.cut(
            vals,
            bins=WIND_ALIGNMENT_BINS,
            labels=WIND_ALIGNMENT_LABELS,
            include_lowest=True,
        ).astype("object")

    return out

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = add_binary_metrics(df)

    out["IsEmptyPrediction"] = False
    out["IsNearEmptyPrediction"] = False
    out["IsSevereFalseNegative"] = False
    out["IsSevereFalsePositive"] = False
    out["IsOversegmented"] = False
    out["IsUndersegmented"] = False
    out["IsSaturatedPrediction"] = False
    out["IsAcceptablePrediction"] = False
    out["PrimaryQualityCategory"] = "Otros"

    if "PredArea" in out.columns:
        out["IsEmptyPrediction"] = out["PredArea"].fillna(0) == 0

    if all(c in out.columns for c in ["PredArea", "GTArea", "AreaRatio"]):
        out["IsNearEmptyPrediction"] = out["PredArea"].fillna(0) < QUALITY_RULES["near_empty_ratio"] * out["GTArea"].fillna(0)
        out["IsOversegmented"] = out["AreaRatio"].fillna(0) > QUALITY_RULES["oversegmentation_ratio"]
        out["IsUndersegmented"] = out["AreaRatio"].fillna(np.inf) < QUALITY_RULES["undersegmentation_ratio"]

    if "Recall" in out.columns:
        out["IsSevereFalseNegative"] = out["Recall"].fillna(0) < QUALITY_RULES["severe_fn_recall"]

    if all(c in out.columns for c in ["Precision", "PredArea", "GTArea"]):
        out["IsSevereFalsePositive"] = (out["Precision"].fillna(1) < QUALITY_RULES["severe_fp_precision"]) & (out["PredArea"].fillna(0) > out["GTArea"].fillna(np.inf))

    if "PredictedPositiveFraction" in out.columns:
        out["IsSaturatedPrediction"] = out["PredictedPositiveFraction"].fillna(0) > QUALITY_RULES["saturation_fraction"]

    if all(c in out.columns for c in ["Dice", "AreaRatio"]):
        tp_ok = out["TP"].fillna(1) > 0 if "TP" in out.columns else True
        out["IsAcceptablePrediction"] = (out["Dice"].fillna(0) >= QUALITY_RULES["acceptable_dice"]) & tp_ok & (out["AreaRatio"].fillna(0) >= QUALITY_RULES["acceptable_area_low"]) & (out["AreaRatio"].fillna(np.inf) <= QUALITY_RULES["acceptable_area_high"])

    category_priority = [
        ("Aceptable", "IsAcceptablePrediction"),
        ("Vacía", "IsEmptyPrediction"),
        ("Saturada", "IsSaturatedPrediction"),
        ("FN severo", "IsSevereFalseNegative"),
        ("FP severo", "IsSevereFalsePositive"),
        ("Sobresegmentada", "IsOversegmented"),
        ("Subsegmentada", "IsUndersegmented"),
        ("Casi vacía", "IsNearEmptyPrediction"),
    ]

    for label, flag in reversed(category_priority):
        out.loc[out[flag].fillna(False), "PrimaryQualityCategory"] = label

    return out


def summary_row_from_test_summary(test_summary: pd.DataFrame, run_row: pd.Series) -> Dict:
    if test_summary.empty:
        row = {}
    else:
        row = test_summary.iloc[0].to_dict()

    result = {
        "ExperimentId": run_row["ExperimentId"],
        "ExperimentOrder": run_row["ExperimentOrder"],
        "RunTag": run_row["RunTag"],
        "FeatureConfig": run_row["FeatureConfig"],
        "Configuracion": spanish_config(run_row["FeatureConfig"]),
        "ModelName": run_row["ModelName"],
        "Modelo": spanish_model(run_row["ModelName"]),
        "ShortLabel": run_row["ShortLabel"],
        "ModelRunId": run_row["ModelRunId"],
    }

    for col in ["BestThreshold", "MeanDice", "MeanIoU", "MeanPrecision", "MeanRecall", "GlobalDice", "GlobalIoU", "Samples", "TP", "FP", "FN", "TN"]:
        if col in row:
            result[col] = row[col]

    return result


# =============================================================================
# Tablas LaTeX
# =============================================================================

def dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    path: Path,
    column_format: Optional[str] = None,
    digits: int = 4,
) -> None:
    work = df.copy()
    for col in work.select_dtypes(include=[np.number]).columns:
        work[col] = work[col].map(lambda x: format_float(x, digits))

    if column_format is None:
        column_format = "l" + "c" * (len(work.columns) - 1)

    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("    \\centering")
    latex.append("    \\small")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append(f"    \\label{{{label}}}")
    latex.append(f"    \\begin{{tabular}}{{{column_format}}}")
    latex.append("        \\toprule")
    latex.append("        " + " & ".join(work.columns) + " \\\\")
    latex.append("        \\midrule")
    for _, row in work.iterrows():
        latex.append("        " + " & ".join(str(row[col]) for col in work.columns) + " \\\\")
    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append("\\end{table}")
    write_text("\n".join(latex) + "\n", path)


# =============================================================================
# Figuras por experimento
# =============================================================================

def plot_experiment_main_metrics(exp_summary: pd.DataFrame, title: str, path: Path) -> None:
    metrics = [
        ("MeanDice", "Dice medio"),
        ("MeanIoU", "IoU medio"),
        ("MeanPrecision", "Precisión"),
        ("MeanRecall", "Recall"),
        ("GlobalDice", "Dice global"),
        ("GlobalIoU", "IoU global"),
    ]
    metrics = [(c, label) for c, label in metrics if c in exp_summary.columns]
    if exp_summary.empty or not metrics:
        return

    d = exp_summary.sort_values("FeatureConfig")
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    for idx, (_, row) in enumerate(d.iterrows()):
        cfg = row["FeatureConfig"]
        values = [row[c] for c, _ in metrics]
        offset = -width / 2 if idx == 0 else width / 2
        ax.bar(
            x + offset,
            values,
            width,
            label=spanish_config(cfg),
            color=CONFIG_COLOR.get(cfg, PALETTE["blue"]),
            edgecolor=PALETTE["navy"],
            linewidth=0.45,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor de métrica")
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=18)
    # Leyenda bajo la figura para evitar solape con títulos largos.
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.16),
        borderaxespad=0.0,
        handlelength=2.4,
        columnspacing=1.4,
    )
    clean_axes(ax)
    savefig(fig, path)


def plot_training_curves(history: pd.DataFrame, title: str, path: Path) -> None:
    if history.empty or "Epoch" not in history.columns:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    specs = [
        ("TrainMeanDice", "Dice entrenamiento", PALETTE["navy"], "--"),
        ("ValidationMeanDice", "Dice validación", PALETTE["navy"], "-"),
        ("TrainMeanIoU", "IoU entrenamiento", PALETTE["cyan"], "--"),
        ("ValidationMeanIoU", "IoU validación", PALETTE["cyan"], "-"),
    ]

    for col, label, color, linestyle in specs:
        if col in history.columns:
            ax.plot(history["Epoch"], history[col], marker="o", linewidth=2.4, linestyle=linestyle, color=color, label=label)

    if "ValidationMeanDice" in history.columns and history["ValidationMeanDice"].notna().any():
        idx = history["ValidationMeanDice"].idxmax()
        best = history.loc[idx]
        ax.scatter(best["Epoch"], best["ValidationMeanDice"], s=95, color=PALETTE["navy"], zorder=5)
        ax.annotate(
            f"{best['ValidationMeanDice']:.3f}\nep.{int(best['Epoch'])}",
            xy=(best["Epoch"], best["ValidationMeanDice"]),
            xytext=(8, -20),
            textcoords="offset points",
            fontsize=9,
            color=PALETTE["navy"],
        )

    ax.set_xlabel("Época")
    ax.set_ylabel("Valor de métrica")
    ax.set_ylim(0, 1)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=18)
    # Leyenda bajo la figura para evitar solape con títulos largos.
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        borderaxespad=0.0,
        handlelength=2.2,
        columnspacing=1.2,
    )
    clean_axes(ax)
    savefig(fig, path)


def plot_loss_curves(history: pd.DataFrame, title: str, path: Path) -> None:
    if history.empty or "Epoch" not in history.columns:
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    if "TrainLoss" in history.columns:
        ax.plot(history["Epoch"], history["TrainLoss"], marker="o", linewidth=2.5, color=PALETTE["navy"], label="Pérdida entrenamiento")

    if "ValidationLoss" in history.columns:
        ax.plot(history["Epoch"], history["ValidationLoss"], marker="o", linewidth=2.5, linestyle="--", color=PALETTE["cyan"], label="Pérdida validación")

    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    clean_axes(ax)
    savefig(fig, path)


def plot_correlation_matrix(df: pd.DataFrame, title: str, path: Path) -> None:
    if df.empty:
        return

    column_map = {
        "Dice": "Dice",
        "IoU": "IoU",
        "Precision": "Precisión",
        "Recall": "Recall",
        "GTArea": "Área real",
        "PredArea": "Área predicha",
        "FP": "FP",
        "FN": "FN",
        "ProbabilityMean": "Prob. media",
        "ProbabilityMax": "Prob. máxima",
        "AreaRatio": "Relación área",
        "FPBurden": "Carga FP",
        "FNBurden": "Carga FN",
    }

    cols = [c for c in column_map if c in df.columns]
    if len(cols) < 2:
        return

    corr = df[cols].corr(numeric_only=True)
    labels = [column_map[c] for c in cols]

    fig, ax = plt.subplots(figsize=(8.3, 7.0))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.55 else PALETTE["text"], fontsize=8.5)

    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.78, label="Correlación")
    savefig(fig, path)


def plot_plume_size_vs_performance(df: pd.DataFrame, title: str, path: Path) -> None:
    if df.empty or "GTArea" not in df.columns or "Dice" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.scatter(df["GTArea"], df["Dice"], s=28, alpha=0.65, color=PALETTE["navy"], label="Dice")
    if "IoU" in df.columns:
        ax.scatter(df["GTArea"], df["IoU"], s=28, alpha=0.55, color=PALETTE["cyan"], label="IoU")

    ax.set_xlabel("Tamaño de pluma en ground truth (píxeles)")
    ax.set_ylabel("Valor de métrica")
    ax.set_ylim(0, 1)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False)
    clean_axes(ax, grid_axis="both")
    savefig(fig, path)


def plot_error_distribution(df: pd.DataFrame, title: str, path: Path) -> None:
    if df.empty:
        return
    if not any(c in df.columns for c in ["FP", "FN"]):
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    if "FP" in df.columns:
        ax.hist(df["FP"].dropna(), bins=35, alpha=0.70, color=PALETTE["cyan"], label="Falsos positivos")
    if "FN" in df.columns:
        ax.hist(df["FN"].dropna(), bins=35, alpha=0.65, color=PALETTE["navy"], label="Falsos negativos")

    ax.set_xlabel("Número de píxeles")
    ax.set_ylabel("Número de muestras")
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False)
    clean_axes(ax)
    savefig(fig, path)


def plot_probability_distribution(df: pd.DataFrame, title: str, path: Path) -> None:
    if df.empty:
        return

    cols = [c for c in ["ProbabilityMean", "ProbabilityMax", "ProbabilityP95"] if c in df.columns]
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    colors = {
        "ProbabilityMean": PALETTE["cyan"],
        "ProbabilityMax": PALETTE["navy"],
        "ProbabilityP95": PALETTE["blue"],
    }

    labels = {
        "ProbabilityMean": "Probabilidad media",
        "ProbabilityMax": "Probabilidad máxima",
        "ProbabilityP95": "Percentil 95",
    }

    for col in cols:
        ax.hist(df[col].dropna(), bins=np.linspace(0, 1, 36), alpha=0.55, color=colors[col], label=labels[col])

    ax.set_xlabel("Probabilidad")
    ax.set_ylabel("Número de muestras")
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False)
    clean_axes(ax)
    savefig(fig, path)


def plot_confusion_two_configs(exp_summary: pd.DataFrame, title: str, path: Path) -> None:
    if exp_summary.empty or not all(c in exp_summary.columns for c in ["TP", "FP", "FN", "TN"]):
        return

    d = exp_summary.sort_values("FeatureConfig")
    n = len(d)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)

    for ax, (_, row) in zip(axes.ravel(), d.iterrows()):
        mat = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=float)
        norm = mat / (mat.sum(axis=1, keepdims=True) + EPS)

        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(spanish_config(row["FeatureConfig"]), color=PALETTE["navy"], fontweight="bold", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred. fondo", "Pred. pluma"], rotation=25, ha="right")
        ax.set_yticklabels(["GT fondo", "GT pluma"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    f"{norm[i, j]:.2f}\n{int(mat[i, j]):,}",
                    ha="center",
                    va="center",
                    color="white" if norm[i, j] > 0.55 else PALETTE["text"],
                    fontsize=8,
                )

    fig.suptitle(title, color=PALETTE["navy"], fontweight="bold", y=1.03)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Fracción normalizada por fila")
    savefig(fig, path)


def plot_fp_fn_burden(exp_summary: pd.DataFrame, title: str, path: Path) -> None:
    if exp_summary.empty or not all(c in exp_summary.columns for c in ["FP", "FN", "TP"]):
        return

    d = add_binary_metrics(exp_summary.copy()).sort_values("FeatureConfig")
    if "GTArea" not in d.columns:
        return

    x = np.arange(len(d))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2, d["FPBurden"], width, color=PALETTE["cyan"], edgecolor=PALETTE["navy"], linewidth=0.4, label="FP / área real")
    ax.bar(x + width / 2, d["FNBurden"], width, color=PALETTE["navy"], edgecolor=PALETTE["navy"], linewidth=0.4, label="FN / área real")

    ax.set_xticks(x)
    ax.set_xticklabels([spanish_config(c) for c in d["FeatureConfig"]])
    ax.set_ylabel("Carga normalizada por área real")
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False)
    clean_axes(ax)
    savefig(fig, path)


# =============================================================================
# Construcción de tablas por experimento
# =============================================================================


# =============================================================================
# Metadatos, fluxrate y sensibilidad al umbral
# =============================================================================

def load_metadata(metadata_csv: Optional[str], project_root: Path, run_tags: Sequence[str]) -> pd.DataFrame:
    """Load optional sample metadata and standardize relevant fields."""
    if not metadata_csv:
        return pd.DataFrame()

    path = Path(metadata_csv)
    if not path.is_absolute():
        path = (project_root / path).resolve()

    if not path.exists():
        print(f"[WARN] MetadataCsv no existe: {path}")
        return pd.DataFrame()

    meta = standardize_columns(read_csv(path), path, run_tags)
    if meta.empty or "SampleId" not in meta.columns:
        print(f"[WARN] MetadataCsv no tiene SampleId válido: {path}")
        return pd.DataFrame()

    # Keep useful and non-duplicated fields.
    wanted = [
        "SampleId", "Fluxrate", "FluxrateStd", "Country", "Site",
        "WindSpeed10m", "WindAlignment", "WindSource",
        "WindDirCos10m", "WindDirSin10m", "WindU10m", "WindV10m",
    ]
    keep = [c for c in wanted if c in meta.columns]
    meta = meta[keep].copy()
    meta["SampleId"] = meta["SampleId"].astype(str)
    meta = meta.drop_duplicates("SampleId", keep="first")
    meta = prepare_physical_features(meta)
    return meta


def merge_metadata(master: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata by SampleId without overwriting existing model metrics."""
    if master.empty or metadata.empty or "SampleId" not in master.columns:
        return master

    out = master.copy()
    out["SampleId"] = out["SampleId"].astype(str)
    metadata = metadata.copy()
    metadata["SampleId"] = metadata["SampleId"].astype(str)

    add_cols = ["SampleId"] + [c for c in metadata.columns if c != "SampleId" and c not in out.columns]
    out = out.merge(metadata[add_cols], on="SampleId", how="left")
    out = prepare_physical_features(out)
    out = add_quality_flags(add_bins(add_binary_metrics(out)))
    return out


def prepare_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived physical variables used in Results."""
    out = df.copy()

    if "Fluxrate" in out.columns:
        out["Fluxrate"] = pd.to_numeric(out["Fluxrate"], errors="coerce")
        out.loc[out["Fluxrate"] <= 0, "Fluxrate"] = np.nan
        out["FluxrateKgH"] = out["Fluxrate"]
        out["FluxrateTH"] = out["Fluxrate"] / 1000.0
        out["LogFluxrate"] = np.log10(out["Fluxrate"].clip(lower=1e-6))

    if "FluxrateStd" in out.columns:
        out["FluxrateStd"] = pd.to_numeric(out["FluxrateStd"], errors="coerce")
        out["FluxrateStdKgH"] = out["FluxrateStd"]

    if "Fluxrate" in out.columns and "FluxrateStd" in out.columns:
        out["FluxrateRelativeUncertainty"] = out["FluxrateStd"] / (out["Fluxrate"] + EPS)
        out["FluxrateUncertaintyPct"] = 100.0 * out["FluxrateRelativeUncertainty"]

    return out


def build_fluxrate_outputs(master: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    """Build fluxrate-specific tables and figures."""
    if master.empty or "Fluxrate" not in master.columns or master["Fluxrate"].notna().sum() < 10:
        write_text(
            "No se generó análisis de fluxrate porque la tabla maestra no contiene Fluxrate suficiente. "
            "Verifique --MetadataCsv con SampleId, ch4_fluxrate y ch4_fluxrate_std.\n",
            dirs["Logs"] / "ResultsFluxrateNoDisponible.md",
        )
        return

    d = prepare_physical_features(add_quality_flags(add_bins(add_binary_metrics(master))))
    d = d[d["Fluxrate"].notna()].copy()

    # Distribution table.
    dist_rows = [{
        "N": len(d.drop_duplicates("SampleId")) if "SampleId" in d.columns else len(d),
        "FluxrateKgHMin": d["Fluxrate"].min(),
        "FluxrateKgHQ1": d["Fluxrate"].quantile(0.25),
        "FluxrateKgHMedian": d["Fluxrate"].median(),
        "FluxrateKgHQ3": d["Fluxrate"].quantile(0.75),
        "FluxrateKgHMax": d["Fluxrate"].max(),
        "FluxrateTHMedian": d["FluxrateTH"].median() if "FluxrateTH" in d.columns else np.nan,
    }]
    write_csv(pd.DataFrame(dist_rows), dirs["Tables"] / "ResultsFluxrateDistribution.csv", digits=4)

    # Metrics by fluxrate range.
    metrics = [c for c in ["Dice", "IoU", "Precision", "Recall", "FalseNegativeRate", "FPBurden", "FNBurden", "AreaRatio"] if c in d.columns]
    keys = [c for c in ["ExperimentId", "ModelName", "Modelo", "FeatureConfig", "ShortLabel", "FluxrateBin"] if c in d.columns]
    if "FluxrateBin" in d.columns and metrics:
        tab = d.dropna(subset=["FluxrateBin"]).groupby(keys, dropna=False)[metrics].agg(["mean", "median", "count"])
        tab.columns = [pascal_case("_".join(col)) for col in tab.columns]
        tab = tab.reset_index()
        write_csv(tab, dirs["Tables"] / "ResultsMetricasPorRangoFluxrate.csv", digits=4)

        latex_cols = [c for c in ["Modelo", "FeatureConfig", "FluxrateBin", "DiceMean", "IoUMean", "PrecisionMean", "RecallMean", "FalseNegativeRateMean", "FPBurdenMean", "FNBurdenMean"] if c in tab.columns]
        if latex_cols:
            tex_tab = tab[latex_cols].copy()
            tex_tab = tex_tab.rename(columns={
                "FeatureConfig": "Configuración",
                "FluxrateBin": "Rango fluxrate",
                "DiceMean": "Dice",
                "IoUMean": "IoU",
                "PrecisionMean": "Precisión",
                "RecallMean": "Recall",
                "FalseNegativeRateMean": "FNR",
                "FPBurdenMean": "FP/área",
                "FNBurdenMean": "FN/área",
            })
            tex_tab["Configuración"] = tex_tab["Configuración"].map(spanish_config)
            dataframe_to_latex_table(
                tex_tab,
                caption="Métricas de segmentación por rango de fluxrate.",
                label="tab:metricas_por_rango_fluxrate",
                path=dirs["Latex"] / "TableMetricasPorRangoFluxrate.tex",
                digits=4,
            )

    # ConfigC - ConfigB deltas by model and sample.
    delta = build_config_delta_table(d)
    if not delta.empty:
        write_csv(delta, dirs["Tables"] / "ResultsDeltaConfigCPorMuestra.csv", digits=4)
        if "FluxrateBin" in delta.columns:
            delta_metrics = [c for c in ["DeltaDice", "DeltaRecall", "DeltaPrecision", "DeltaFPBurden", "DeltaFNBurden"] if c in delta.columns]
            delta_tab = delta.groupby(["Modelo", "ModelName", "FluxrateBin"], dropna=False)[delta_metrics].agg(["mean", "median", "count"])
            delta_tab.columns = [pascal_case("_".join(col)) for col in delta_tab.columns]
            delta_tab = delta_tab.reset_index()
            write_csv(delta_tab, dirs["Tables"] / "ResultsDeltaConfigCPorRangoFluxrate.csv", digits=4)

            tex_cols = [c for c in ["Modelo", "FluxrateBin", "DeltaDiceMean", "DeltaRecallMean", "DeltaPrecisionMean", "DeltaFPBurdenMean", "DeltaFNBurdenMean"] if c in delta_tab.columns]
            if tex_cols:
                tex_delta = delta_tab[tex_cols].rename(columns={
                    "FluxrateBin": "Rango fluxrate",
                    "DeltaDiceMean": "$\\Delta$Dice",
                    "DeltaRecallMean": "$\\Delta$Recall",
                    "DeltaPrecisionMean": "$\\Delta$Prec.",
                    "DeltaFPBurdenMean": "$\\Delta$FP/área",
                    "DeltaFNBurdenMean": "$\\Delta$FN/área",
                })
                dataframe_to_latex_table(
                    tex_delta,
                    caption="Diferencia media entre Configuración C y Configuración B por rango de fluxrate.",
                    label="tab:delta_configc_por_rango_fluxrate",
                    path=dirs["Latex"] / "TableDeltaConfigCPorRangoFluxrate.tex",
                    digits=4,
                )

    # Correlations.
    corr_rows = []
    for (model, cfg), g in d.groupby(["ModelName", "FeatureConfig"]):
        for metric in metrics:
            tmp = g[["Fluxrate", metric]].dropna()
            if len(tmp) >= 10:
                corr_rows.append({
                    "Modelo": spanish_model(model),
                    "Configuración": spanish_config(cfg),
                    "Métrica": metric,
                    "N": len(tmp),
                    "Spearman": tmp["Fluxrate"].corr(tmp[metric], method="spearman"),
                    "Pearson": tmp["Fluxrate"].corr(tmp[metric], method="pearson"),
                })
    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        write_csv(corr, dirs["Tables"] / "ResultsCorrelacionesFluxrate.csv", digits=4)
        dataframe_to_latex_table(
            corr,
            caption="Correlaciones entre fluxrate y métricas de desempeño.",
            label="tab:correlaciones_fluxrate",
            path=dirs["Latex"] / "TableCorrelacionesFluxrate.tex",
            digits=4,
        )

    # Uncertainty analysis.
    if "FluxrateRelativeUncertainty" in d.columns and d["FluxrateRelativeUncertainty"].notna().sum() >= 10:
        if "FluxrateUncertaintyBin" not in d.columns or d["FluxrateUncertaintyBin"].isna().all():
            d = add_bins(d)
        u_metrics = [c for c in ["Dice", "Recall", "FalseNegativeRate", "FPBurden", "FNBurden"] if c in d.columns]
        if "FluxrateUncertaintyBin" in d.columns and u_metrics:
            u_tab = d.dropna(subset=["FluxrateUncertaintyBin"]).groupby(
                ["Modelo", "ModelName", "FeatureConfig", "FluxrateUncertaintyBin"], dropna=False
            )[u_metrics].agg(["mean", "median", "count"])
            u_tab.columns = [pascal_case("_".join(col)) for col in u_tab.columns]
            u_tab = u_tab.reset_index()
            write_csv(u_tab, dirs["Tables"] / "ResultsMetricasPorIncertidumbreFluxrate.csv", digits=4)

            tex_cols = [c for c in ["Modelo", "FeatureConfig", "FluxrateUncertaintyBin", "DiceMean", "RecallMean", "FalseNegativeRateMean", "FPBurdenMean", "FNBurdenMean"] if c in u_tab.columns]
            if tex_cols:
                tex_u = u_tab[tex_cols].rename(columns={
                    "FeatureConfig": "Configuración",
                    "FluxrateUncertaintyBin": "Incertidumbre",
                    "DiceMean": "Dice",
                    "RecallMean": "Recall",
                    "FalseNegativeRateMean": "FNR",
                    "FPBurdenMean": "FP/área",
                    "FNBurdenMean": "FN/área",
                })
                tex_u["Configuración"] = tex_u["Configuración"].map(spanish_config)
                dataframe_to_latex_table(
                    tex_u,
                    caption="Métricas por rango de incertidumbre relativa del fluxrate.",
                    label="tab:metricas_por_incertidumbre_fluxrate",
                    path=dirs["Latex"] / "TableMetricasPorIncertidumbreFluxrate.tex",
                    digits=4,
                )

    # Figures.
    plot_fluxrate_distribution(d, dirs["Figures"] / "FigureFluxrateDistribucion.png")
    plot_fluxrate_factor_bar(d, "FluxrateBin", "Dice", "Dice por rango de fluxrate", dirs["Figures"] / "FigureDicePorRangoFluxrate.png", "Dice medio")
    plot_fluxrate_factor_bar(d, "FluxrateBin", "Recall", "Recall por rango de fluxrate", dirs["Figures"] / "FigureRecallPorRangoFluxrate.png", "Recall medio")
    plot_fluxrate_factor_bar(d, "FluxrateBin", "FalseNegativeRate", "Falsos negativos por rango de fluxrate", dirs["Figures"] / "FigureFnRatePorRangoFluxrate.png", "Tasa media de falsos negativos")
    plot_fluxrate_scatter(d, "LogFluxrate", "Dice", "Fluxrate vs Dice", dirs["Figures"] / "FigureFluxrateVsDice.png", "log10(fluxrate kg/h)", "Dice")
    plot_fluxrate_scatter(d, "LogFluxrate", "Recall", "Fluxrate vs recall", dirs["Figures"] / "FigureFluxrateVsRecall.png", "log10(fluxrate kg/h)", "Recall")
    plot_fluxrate_scatter(d, "LogFluxrate", "FalseNegativeRate", "Fluxrate vs falsos negativos", dirs["Figures"] / "FigureFluxrateVsFnRate.png", "log10(fluxrate kg/h)", "Tasa de falsos negativos")
    if not delta.empty:
        plot_delta_by_factor(delta, "FluxrateBin", "DeltaDice", "Diferencia Configuración C - B por fluxrate", dirs["Figures"] / "FigureDeltaDiceConfigCPorRangoFluxrate.png", "$\\Delta$Dice")
        plot_delta_by_factor(delta, "FluxrateBin", "DeltaRecall", "Diferencia de Recall C - B por fluxrate", dirs["Figures"] / "FigureDeltaRecallConfigCPorRangoFluxrate.png", "$\\Delta$Recall")
    if "FluxrateUncertaintyBin" in d.columns:
        plot_fluxrate_factor_bar(d, "FluxrateUncertaintyBin", "Dice", "Dice por incertidumbre relativa del fluxrate", dirs["Figures"] / "FigureDicePorIncertidumbreFluxrate.png", "Dice medio")


def build_config_delta_table(master: pd.DataFrame) -> pd.DataFrame:
    """Compute ConfigC - ConfigB deltas for same model and SampleId."""
    needed = {"ModelName", "FeatureConfig", "SampleId"}
    if master.empty or not needed.issubset(master.columns):
        return pd.DataFrame()

    metrics = [c for c in ["Dice", "Recall", "Precision", "FPBurden", "FNBurden", "FalseNegativeRate", "AreaRatio"] if c in master.columns]
    extra = [c for c in ["Fluxrate", "FluxrateTH", "LogFluxrate", "FluxrateBin", "FluxrateRelativeUncertainty", "FluxrateUncertaintyBin", "GTArea", "PlumeSizeBin", "WindSpeed10m", "WindSpeedBin", "WindAlignment", "WindAlignmentBin"] if c in master.columns]
    cols = ["ModelName", "Modelo", "SampleId", "FeatureConfig"] + metrics + extra
    d = master[cols].copy()
    b = d[d["FeatureConfig"] == "ConfigB"].copy()
    c = d[d["FeatureConfig"] == "ConfigC"].copy()
    if b.empty or c.empty:
        return pd.DataFrame()

    keys = ["ModelName", "SampleId"]
    b = b.rename(columns={m: f"{m}_B" for m in metrics})
    c = c.rename(columns={m: f"{m}_C" for m in metrics})
    merged = b.merge(c, on=keys, suffixes=("_Brow", "_Crow"))
    if merged.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "ModelName": merged["ModelName"],
        "Modelo": merged.get("Modelo_Brow", merged["ModelName"].map(spanish_model)),
        "SampleId": merged["SampleId"],
    })
    for e in extra:
        col = f"{e}_Brow"
        if col in merged.columns:
            out[e] = merged[col]
        elif e in merged.columns:
            out[e] = merged[e]
    for m in metrics:
        out[f"Delta{m}"] = merged[f"{m}_C"] - merged[f"{m}_B"]
    return out


def plot_fluxrate_distribution(df: pd.DataFrame, path: Path) -> None:
    if df.empty or "FluxrateTH" not in df.columns:
        return
    vals = df.drop_duplicates("SampleId")["FluxrateTH"].dropna() if "SampleId" in df.columns else df["FluxrateTH"].dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(vals, bins=35, color=PALETTE["blue"], edgecolor=PALETTE["navy"], alpha=0.82)
    ax.set_xlabel("Fluxrate (t/h)")
    ax.set_ylabel("Número de muestras")
    ax.set_title("Distribución de tasas de emisión en test", color=PALETTE["navy"], fontweight="bold", pad=12)
    clean_axes(ax)
    savefig(fig, path)


def plot_fluxrate_factor_bar(df: pd.DataFrame, factor: str, metric: str, title: str, path: Path, ylabel: str) -> None:
    if df.empty or factor not in df.columns or metric not in df.columns:
        return
    grouped = df.dropna(subset=[factor, metric]).groupby(["ModelName", "FeatureConfig", factor])[metric].mean().reset_index()
    if grouped.empty:
        return
    levels = [x for x in ["Bajo", "Medio", "Alto", "Baja", "Media", "Alta", "Pequeña", "Grande"] if x in set(grouped[factor].astype(str))]
    if not levels:
        levels = list(grouped[factor].astype(str).dropna().unique())
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    x = np.arange(len(levels))
    width = 0.10
    combos = grouped[["ModelName", "FeatureConfig"]].drop_duplicates().sort_values(["ModelName", "FeatureConfig"]).values.tolist()
    offsets = (np.arange(len(combos)) - (len(combos)-1)/2) * width
    for i, (model, cfg) in enumerate(combos):
        vals = []
        for level in levels:
            tmp = grouped[(grouped["ModelName"] == model) & (grouped["FeatureConfig"] == cfg) & (grouped[factor].astype(str) == str(level))]
            vals.append(tmp[metric].iloc[0] if len(tmp) else np.nan)
        color = MODEL_COLOR.get(model, CONFIG_COLOR.get(cfg, PALETTE["blue"]))
        hatch = "" if cfg == "ConfigB" else "//"
        ax.bar(x + offsets[i], vals, width, label=short_label(model, cfg), color=color, hatch=hatch, edgecolor=PALETTE["navy"], linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    clean_axes(ax)
    savefig(fig, path)


def plot_fluxrate_scatter(df: pd.DataFrame, x: str, y: str, title: str, path: Path, xlabel: str, ylabel: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for (model, cfg), g in df.dropna(subset=[x, y]).groupby(["ModelName", "FeatureConfig"]):
        color = MODEL_COLOR.get(model, CONFIG_COLOR.get(cfg, PALETTE["blue"]))
        marker = "o" if cfg == "ConfigB" else "s"
        ax.scatter(g[x], g[y], s=18, alpha=0.42, color=color, marker=marker, edgecolors="none", label=short_label(model, cfg))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    clean_axes(ax, grid_axis="both")
    savefig(fig, path)


def plot_delta_by_factor(delta: pd.DataFrame, factor: str, metric: str, title: str, path: Path, ylabel: str) -> None:
    if delta.empty or factor not in delta.columns or metric not in delta.columns:
        return
    grouped = delta.dropna(subset=[factor, metric]).groupby(["ModelName", "Modelo", factor])[metric].mean().reset_index()
    if grouped.empty:
        return
    levels = [x for x in ["Bajo", "Medio", "Alto", "Baja", "Media", "Alta", "Pequeña", "Grande"] if x in set(grouped[factor].astype(str))]
    if not levels:
        levels = list(grouped[factor].astype(str).dropna().unique())
    models = list(grouped["ModelName"].drop_duplicates())
    x = np.arange(len(levels))
    width = min(0.18, 0.75 / max(1, len(models)))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for i, model in enumerate(models):
        vals = []
        for level in levels:
            tmp = grouped[(grouped["ModelName"] == model) & (grouped[factor].astype(str) == str(level))]
            vals.append(tmp[metric].iloc[0] if len(tmp) else np.nan)
        ax.bar(x + (i - (len(models)-1)/2)*width, vals, width, label=spanish_model(model), color=MODEL_COLOR.get(model, PALETTE["blue"]), edgecolor=PALETTE["navy"], linewidth=0.35)
    ax.axhline(0, color=PALETTE["muted"], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False)
    clean_axes(ax)
    savefig(fig, path)


def collect_threshold_tables(inventory: pd.DataFrame, run_tags: Sequence[str]) -> pd.DataFrame:
    """Collect threshold sweep tables if they exist in model Metrics folders."""
    frames = []
    candidate_keywords = ["threshold", "umbr", "sweep"]
    for _, run in inventory.iterrows():
        metrics_dir = Path(run["MetricsDirectory"])
        if not metrics_dir.exists():
            continue
        for path in metrics_dir.glob("*.csv"):
            if not any(k in path.name.lower() for k in candidate_keywords):
                continue
            raw = read_csv(path)
            if raw.empty:
                continue
            df = standardize_columns(raw, path, run_tags)
            # Recover threshold column if BestThreshold swallowed it.
            if "Threshold" not in df.columns:
                for c in raw.columns:
                    if normalize(c) in {"threshold", "evalthreshold"}:
                        df["Threshold"] = pd.to_numeric(raw[c], errors="coerce")
                        break
            if "Threshold" not in df.columns and "BestThreshold" in df.columns and len(df) > 1:
                df["Threshold"] = df["BestThreshold"]
            if "Threshold" not in df.columns:
                continue
            if not any(c in df.columns for c in ["MeanDice", "MeanPrecision", "MeanRecall", "TP", "FP", "FN", "TN"]):
                continue
            df["RunTag"] = run["RunTag"]
            df["FeatureConfig"] = run["FeatureConfig"]
            df["ModelName"] = run["ModelName"]
            df["Modelo"] = run["SpanishModel"]
            df["ShortLabel"] = run["ShortLabel"]
            df["ExperimentId"] = run["ExperimentId"]
            df["SourcePath"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = add_binary_metrics(out)
    return out


def build_threshold_outputs(inventory: pd.DataFrame, run_tags: Sequence[str], dirs: Dict[str, Path]) -> None:
    t = collect_threshold_tables(inventory, run_tags)
    if t.empty:
        write_text(
            "No se encontraron tablas de sensibilidad al umbral. "
            "Busque o exporte CSVs por modelo con columnas Threshold, MeanDice, MeanPrecision, MeanRecall, TP, FP, FN, TN.\n",
            dirs["Logs"] / "ResultsThresholdsNoDisponibles.md",
        )
        return

    write_csv(t, dirs["Tables"] / "ResultsMetricasPorUmbral.csv", digits=4)

    # Best thresholds table by MeanDice.
    if "MeanDice" in t.columns:
        best = t.sort_values("MeanDice", ascending=False).drop_duplicates(["ModelName", "FeatureConfig"], keep="first")
        best_tab = best[["ExperimentId", "Modelo", "FeatureConfig", "Threshold", "MeanDice", "MeanPrecision", "MeanRecall"]].copy()
        best_tab = best_tab.rename(columns={
            "FeatureConfig": "Configuración",
            "Threshold": "Umbral",
            "MeanDice": "Dice medio",
            "MeanPrecision": "Precisión",
            "MeanRecall": "Recall",
        })
        best_tab["Configuración"] = best_tab["Configuración"].map(spanish_config)
        write_csv(best_tab, dirs["Tables"] / "ResultsUmbralesOptimos.csv", digits=4)
        dataframe_to_latex_table(
            best_tab,
            caption="Umbral óptimo por modelo y configuración según Dice medio.",
            label="tab:umbrales_optimos",
            path=dirs["Latex"] / "TableUmbralesOptimos.tex",
            digits=4,
        )

    plot_threshold_metric(t, "MeanDice", "Dice medio según umbral", dirs["Figures"] / "FigureCurvaDiceUmbralGlobal.png", "Dice medio")
    plot_threshold_metric(t, "MeanRecall", "Recall según umbral", dirs["Figures"] / "FigureCurvaRecallUmbralGlobal.png", "Recall medio")
    plot_threshold_metric(t, "MeanPrecision", "Precisión según umbral", dirs["Figures"] / "FigureCurvaPrecisionUmbralGlobal.png", "Precisión media")
    if "FPBurden" in t.columns:
        plot_threshold_metric(t, "FPBurden", "Carga de falsos positivos según umbral", dirs["Figures"] / "FigureCurvaFpBurdenUmbralGlobal.png", "FP / área real")
    if "FNBurden" in t.columns:
        plot_threshold_metric(t, "FNBurden", "Carga de falsos negativos según umbral", dirs["Figures"] / "FigureCurvaFnBurdenUmbralGlobal.png", "FN / área real")
    plot_precision_recall_curve(t, dirs["Figures"] / "FigureCurvaPrecisionRecallGlobal.png")
    plot_roc_curve_optional(t, dirs["Figures"] / "FigureCurvaRocGlobal.png")


def plot_threshold_metric(df: pd.DataFrame, metric: str, title: str, path: Path, ylabel: str) -> None:
    if df.empty or "Threshold" not in df.columns or metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for (model, cfg), g in df.dropna(subset=["Threshold", metric]).groupby(["ModelName", "FeatureConfig"]):
        g = g.sort_values("Threshold")
        color = MODEL_COLOR.get(model, PALETTE["blue"])
        linestyle = "-" if cfg == "ConfigB" else "--"
        marker = "o" if cfg == "ConfigB" else "s"
        ax.plot(g["Threshold"], g[metric], marker=marker, linewidth=2.2, linestyle=linestyle, color=color, label=short_label(model, cfg))
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.set_xticks(sorted(df["Threshold"].dropna().unique()))
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    clean_axes(ax)
    savefig(fig, path)


def plot_precision_recall_curve(df: pd.DataFrame, path: Path) -> None:
    if df.empty or "MeanPrecision" not in df.columns or "MeanRecall" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for (model, cfg), g in df.dropna(subset=["MeanPrecision", "MeanRecall"]).groupby(["ModelName", "FeatureConfig"]):
        g = g.sort_values("Threshold") if "Threshold" in g.columns else g
        color = MODEL_COLOR.get(model, PALETTE["blue"])
        linestyle = "-" if cfg == "ConfigB" else "--"
        ax.plot(g["MeanRecall"], g["MeanPrecision"], marker="o", linewidth=2.2, linestyle=linestyle, color=color, label=short_label(model, cfg))
    ax.set_xlabel("Recall medio")
    ax.set_ylabel("Precisión media")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Curva Precisión--Recall", color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    clean_axes(ax, grid_axis="both")
    savefig(fig, path)


def plot_roc_curve_optional(df: pd.DataFrame, path: Path) -> None:
    if df.empty or not all(c in df.columns for c in ["FalsePositiveRate", "Recall"]):
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for (model, cfg), g in df.dropna(subset=["FalsePositiveRate", "Recall"]).groupby(["ModelName", "FeatureConfig"]):
        g = g.sort_values("FalsePositiveRate")
        color = MODEL_COLOR.get(model, PALETTE["blue"])
        linestyle = "-" if cfg == "ConfigB" else "--"
        ax.plot(g["FalsePositiveRate"], g["Recall"], marker="o", linewidth=2.0, linestyle=linestyle, color=color, label=short_label(model, cfg))
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Recall / TPR")
    ax.set_xlim(0, min(1, max(0.05, df["FalsePositiveRate"].max() * 1.1)))
    ax.set_ylim(0, 1)
    ax.set_title("Curva ROC complementaria", color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    clean_axes(ax, grid_axis="both")
    savefig(fig, path)


def build_model_summary(inventory: pd.DataFrame, run_tags: Sequence[str]) -> pd.DataFrame:
    rows = []
    for _, run in inventory.iterrows():
        tables = load_run_tables(run, run_tags)
        test = tables["TestMetricsSummary"]
        if test.empty:
            continue
        rows.append(summary_row_from_test_summary(test, run))

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary = standardize_columns(summary, None, run_tags)
    summary = add_binary_metrics(summary)
    summary = summary.sort_values(["ExperimentOrder", "FeatureConfig"]).reset_index(drop=True)
    return summary


def build_master_by_sample(inventory: pd.DataFrame, run_tags: Sequence[str]) -> pd.DataFrame:
    frames = []
    for _, run in inventory.iterrows():
        tables = load_run_tables(run, run_tags)
        by_sample = tables["TestMetricsBySample"]
        if by_sample.empty:
            continue

        by_sample = by_sample.copy()
        by_sample["RunTag"] = run["RunTag"]
        by_sample["FeatureConfig"] = run["FeatureConfig"]
        by_sample["ModelName"] = run["ModelName"]
        by_sample["Modelo"] = run["SpanishModel"]
        by_sample["ShortLabel"] = run["ShortLabel"]
        by_sample["ModelRunId"] = run["ModelRunId"]
        by_sample["ExperimentId"] = run["ExperimentId"]
        by_sample["ExperimentOrder"] = run["ExperimentOrder"]
        frames.append(by_sample)

    if not frames:
        return pd.DataFrame()

    master = pd.concat(frames, ignore_index=True, sort=False)
    master = standardize_columns(master, None, run_tags)
    master = add_quality_flags(add_bins(add_binary_metrics(master)))
    return master


def experiment_summary_tables(exp_summary: pd.DataFrame, slug: str, title: str, dirs: Dict[str, Path]) -> None:
    if exp_summary.empty:
        return

    metrics_cols = [
        ("Configuración", "Configuracion"),
        ("Umbral", "BestThreshold"),
        ("Dice medio", "MeanDice"),
        ("IoU medio", "MeanIoU"),
        ("Precisión", "MeanPrecision"),
        ("Recall", "MeanRecall"),
        ("Dice global", "GlobalDice"),
        ("IoU global", "GlobalIoU"),
    ]

    tab = pd.DataFrame()
    for label, col in metrics_cols:
        if col in exp_summary.columns:
            tab[label] = exp_summary[col]

    write_csv(tab, dirs["Tables"] / f"Results{slug}MetricasPrincipales.csv", digits=4)
    dataframe_to_latex_table(
        tab,
        caption=f"Métricas principales de {title.lower()} para Configuración B y Configuración C.",
        label=f"tab:{slug.lower()}_metricas_principales",
        path=dirs["Latex"] / f"Table{slug}MetricasPrincipales.tex",
        digits=4,
    )

    if all(c in exp_summary.columns for c in ["TP", "FP", "FN", "TN"]):
        conf = add_binary_metrics(exp_summary.copy())
        conf_tab = pd.DataFrame({
            "Configuración": conf["Configuracion"],
            "TP": conf["TP"],
            "FP": conf["FP"],
            "FN": conf["FN"],
            "TN": conf["TN"],
            "FPR": conf["FalsePositiveRate"],
            "FNR": conf["FalseNegativeRate"],
            "MCC": conf["MCC"],
            "Balanced Acc.": conf["BalancedAccuracy"],
            "FP / área real": conf["FPBurden"],
            "FN / área real": conf["FNBurden"],
        })
        write_csv(conf_tab, dirs["Tables"] / f"Results{slug}FalsosPositivosNegativos.csv", digits=4)
        dataframe_to_latex_table(
            conf_tab,
            caption=f"Análisis de falsos positivos y falsos negativos para {title.lower()}.",
            label=f"tab:{slug.lower()}_fp_fn",
            path=dirs["Latex"] / f"Table{slug}FalsosPositivosNegativos.tex",
            digits=4,
        )


def build_experiment_outputs(
    model_name: str,
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    master: pd.DataFrame,
    run_tags: Sequence[str],
    dirs: Dict[str, Path],
) -> None:
    if model_name not in EXPERIMENTS:
        raise ValueError(f"Experimento no soportado: {model_name}")

    meta = EXPERIMENTS[model_name]
    slug = meta["Slug"]
    title = meta["Title"]

    exp_summary = summary[summary["ModelName"] == model_name].copy()
    exp_sample = master[master["ModelName"] == model_name].copy() if not master.empty else pd.DataFrame()

    if exp_summary.empty:
        print(f"[WARN] No hay resumen para {title}")
        return

    experiment_summary_tables(exp_summary, slug, title, dirs)

    plot_experiment_main_metrics(
        exp_summary,
        title=f"{title}: métricas principales",
        path=dirs["Figures"] / f"Figure{slug}MetricasPrincipales.png",
    )

    plot_confusion_two_configs(
        exp_summary,
        title=f"{title}: matrices de confusión",
        path=dirs["Figures"] / f"Figure{slug}MatricesConfusion.png",
    )

    plot_fp_fn_burden(
        exp_summary,
        title=f"{title}: carga de FP y FN",
        path=dirs["Figures"] / f"Figure{slug}CargaFpFn.png",
    )

    # Figuras específicas por configuración.
    exp_runs = inventory[inventory["ModelName"] == model_name].sort_values("FeatureConfig")
    for _, run in exp_runs.iterrows():
        tables = load_run_tables(run, run_tags)
        cfg = run["FeatureConfig"]
        cfg_short = "ConfigB" if cfg == "ConfigB" else "ConfigC"
        cfg_spanish = spanish_config(cfg)

        history = tables["TrainingHistory"]
        test_by_sample = tables["TestMetricsBySample"]
        if not test_by_sample.empty:
            test_by_sample = add_quality_flags(add_bins(add_binary_metrics(test_by_sample)))

        plot_training_curves(
            history,
            title=f"{title} · {cfg_spanish}\nCurvas de entrenamiento",
            path=dirs["Figures"] / f"Figure{slug}CurvasEntrenamiento{cfg_short}.png",
        )

        plot_loss_curves(
            history,
            title=f"{title} · {cfg_spanish}\nEvolución de la pérdida",
            path=dirs["Figures"] / f"Figure{slug}EvolucionPerdida{cfg_short}.png",
        )

        plot_correlation_matrix(
            test_by_sample,
            title=f"{title} · {cfg_spanish}: matriz de correlación",
            path=dirs["Figures"] / f"Figure{slug}MatrizCorrelacion{cfg_short}.png",
        )

        plot_plume_size_vs_performance(
            test_by_sample,
            title=f"{title} · {cfg_spanish}: tamaño de pluma vs desempeño",
            path=dirs["Figures"] / f"Figure{slug}TamanoPlumaVsDesempeno{cfg_short}.png",
        )

        plot_error_distribution(
            test_by_sample,
            title=f"{title} · {cfg_spanish}: distribución de errores FP/FN",
            path=dirs["Figures"] / f"Figure{slug}DistribucionErroresFpFn{cfg_short}.png",
        )

        plot_probability_distribution(
            test_by_sample,
            title=f"{title} · {cfg_spanish}: distribución de probabilidad predicha",
            path=dirs["Figures"] / f"Figure{slug}DistribucionProbabilidadPredicha{cfg_short}.png",
        )

    # Distribuciones comparadas si existe tabla maestra.
    if not exp_sample.empty:
        plot_distribution_by_config(
            exp_sample,
            metric="Dice",
            title=f"{title}: distribución de Dice por muestra",
            path=dirs["Figures"] / f"Figure{slug}DistribucionDicePorMuestra.png",
            ylabel="Dice",
        )
        plot_distribution_by_config(
            exp_sample,
            metric="FalseNegativeRate",
            title=f"{title}: distribución de falsos negativos",
            path=dirs["Figures"] / f"Figure{slug}DistribucionFnRate.png",
            ylabel="Tasa de falsos negativos",
        )
        plot_distribution_by_config(
            exp_sample,
            metric="AreaRatio",
            title=f"{title}: relación área predicha / área real",
            path=dirs["Figures"] / f"Figure{slug}DistribucionAreaRatio.png",
            ylabel="Área predicha / área real",
        )


def plot_distribution_by_config(df: pd.DataFrame, metric: str, title: str, path: Path, ylabel: str) -> None:
    if df.empty or metric not in df.columns or "FeatureConfig" not in df.columns:
        return

    configs = [c for c in ["ConfigB", "ConfigC"] if c in set(df["FeatureConfig"])]
    if not configs:
        return

    data = [df[df["FeatureConfig"] == cfg][metric].dropna().values for cfg in configs]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    box = ax.boxplot(data, patch_artist=True, labels=[spanish_config(c) for c in configs], showfliers=False)

    for patch, cfg in zip(box["boxes"], configs):
        patch.set(facecolor=CONFIG_COLOR.get(cfg, PALETTE["blue"]), alpha=0.45, edgecolor=PALETTE["navy"], linewidth=1.0)
    for median in box["medians"]:
        median.set(color=PALETTE["navy"], linewidth=1.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    clean_axes(ax)
    savefig(fig, path)


# =============================================================================
# Análisis por factores y síntesis
# =============================================================================

def build_factors(master: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    if master.empty:
        msg = (
            "No se generaron análisis por factores porque no existe una tabla completa "
            "TestMetricsBySample para los experimentos seleccionados."
        )
        write_text(msg + "\n", dirs["Logs"] / "ResultsFactoresNoDisponibles.md")
        print(f"[WARN] {msg}")
        return

    master = prepare_physical_features(add_quality_flags(add_bins(add_binary_metrics(master))))

    # Fluxrate and uncertainty outputs.
    build_fluxrate_outputs(master, dirs)

    # Tamaño de pluma.
    if "PlumeSizeBin" in master.columns:
        plot_factor_bar(
            master,
            factor="PlumeSizeBin",
            metric="Dice",
            title="Dice por tamaño de pluma",
            path=dirs["Figures"] / "FigureDicePorTamanoPluma.png",
            ylabel="Dice medio",
        )
        plot_factor_bar(
            master,
            factor="PlumeSizeBin",
            metric="FalseNegativeRate",
            title="Falsos negativos por tamaño de pluma",
            path=dirs["Figures"] / "FigureFnRatePorTamanoPluma.png",
            ylabel="Tasa media de falsos negativos",
        )

    # Viento.
    if "WindAlignmentBin" in master.columns:
        plot_factor_bar(
            master,
            factor="WindAlignmentBin",
            metric="Dice",
            title="Dice por alineación viento-pluma",
            path=dirs["Figures"] / "FigureDicePorAlineacionViento.png",
            ylabel="Dice medio",
        )
    if "WindSpeedBin" in master.columns:
        plot_factor_bar(
            master,
            factor="WindSpeedBin",
            metric="Dice",
            title="Dice por velocidad del viento",
            path=dirs["Figures"] / "FigureDicePorVelocidadViento.png",
            ylabel="Dice medio",
        )

def plot_scatter_factor(df: pd.DataFrame, x: str, y: str, title: str, path: Path, xlabel: str, ylabel: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for cfg, g in df.dropna(subset=[x, y]).groupby("FeatureConfig"):
        ax.scatter(
            g[x],
            g[y],
            s=24,
            alpha=0.55,
            color=CONFIG_COLOR.get(cfg, PALETTE["blue"]),
            label=spanish_config(cfg),
            edgecolors="none",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False)
    clean_axes(ax, grid_axis="both")
    savefig(fig, path)


def plot_factor_bar(df: pd.DataFrame, factor: str, metric: str, title: str, path: Path, ylabel: str) -> None:
    if df.empty or factor not in df.columns or metric not in df.columns:
        return

    grouped = (
        df.dropna(subset=[factor, metric])
        .groupby(["FeatureConfig", factor])[metric]
        .mean()
        .reset_index()
    )

    if grouped.empty:
        return

    levels = list(grouped[factor].dropna().unique())
    configs = [c for c in ["ConfigB", "ConfigC"] if c in set(grouped["FeatureConfig"])]

    x = np.arange(len(levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for idx, cfg in enumerate(configs):
        vals = []
        for level in levels:
            tmp = grouped[(grouped["FeatureConfig"] == cfg) & (grouped[factor] == level)]
            vals.append(tmp[metric].iloc[0] if len(tmp) else np.nan)
        offset = -width / 2 if idx == 0 else width / 2
        ax.bar(
            x + offset,
            vals,
            width,
            label=spanish_config(cfg),
            color=CONFIG_COLOR.get(cfg, PALETTE["blue"]),
            edgecolor=PALETTE["navy"],
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.legend(frameon=False)
    clean_axes(ax)
    savefig(fig, path)


def build_summary_outputs(summary: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    if summary.empty:
        return

    ranking = summary.sort_values("MeanDice", ascending=False).copy() if "MeanDice" in summary.columns else summary.copy()

    cols = [
        "ExperimentId", "Modelo", "Configuracion", "BestThreshold",
        "MeanDice", "MeanIoU", "MeanPrecision", "MeanRecall", "GlobalDice", "GlobalIoU"
    ]
    cols = [c for c in cols if c in ranking.columns]
    final_tab = ranking[cols].copy()

    write_csv(final_tab, dirs["Tables"] / "ResultsResumenGlobalMetricas.csv", digits=4)
    dataframe_to_latex_table(
        final_tab.rename(columns={
            "ExperimentId": "Exp.",
            "Modelo": "Modelo",
            "Configuracion": "Configuración",
            "BestThreshold": "Umbral",
            "MeanDice": "Dice medio",
            "MeanIoU": "IoU medio",
            "MeanPrecision": "Precisión",
            "MeanRecall": "Recall",
            "GlobalDice": "Dice global",
            "GlobalIoU": "IoU global",
        }),
        caption="Resumen global de métricas para los experimentos finales.",
        label="tab:resumen_global_metricas",
        path=dirs["Latex"] / "TableResumenGlobalMetricas.tex",
        digits=4,
    )

    plot_global_ranking(summary, "MeanDice", "Ranking global por Dice medio", dirs["Figures"] / "FigureRankingGlobalDiceMedio.png")
    plot_global_ranking(summary, "GlobalDice", "Ranking global por Dice global", dirs["Figures"] / "FigureRankingGlobalDiceGlobal.png")
    plot_global_precision_recall(summary, dirs["Figures"] / "FigurePrecisionRecallGlobal.png")


def plot_global_ranking(summary: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    if summary.empty or metric not in summary.columns:
        return

    d = summary.dropna(subset=[metric]).copy().sort_values(metric, ascending=True)
    labels = [short_label(m, c) for m, c in zip(d["ModelName"], d["FeatureConfig"])]

    fig, ax = plt.subplots(figsize=(8.6, max(4.2, 0.45 * len(d))))
    colors = [MODEL_COLOR.get(m, PALETTE["blue"]) for m in d["ModelName"]]

    ax.barh(labels, d[metric], color=colors, edgecolor=PALETTE["navy"], linewidth=0.4)
    ax.set_xlabel(metric.replace("Mean", "").replace("Global", "Global "))
    ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=12)
    ax.set_xlim(0, min(1.0, max(0.7, d[metric].max() * 1.08)))
    clean_axes(ax)
    savefig(fig, path)


def plot_global_precision_recall(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty or "MeanPrecision" not in summary.columns or "MeanRecall" not in summary.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    for _, row in summary.dropna(subset=["MeanPrecision", "MeanRecall"]).iterrows():
        color = MODEL_COLOR.get(row["ModelName"], PALETTE["blue"])
        marker = "o" if row["FeatureConfig"] == "ConfigB" else "s"
        ax.scatter(row["MeanRecall"], row["MeanPrecision"], s=95, color=color, marker=marker, edgecolor=PALETTE["navy"], linewidth=0.7)
        ax.annotate(
            short_label(row["ModelName"], row["FeatureConfig"]),
            (row["MeanRecall"], row["MeanPrecision"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=PALETTE["text"],
        )

    ax.set_xlabel("Recall medio")
    ax.set_ylabel("Precisión media")
    ax.set_xlim(0.625, 0.75)
    ax.set_ylim(0.625, 0.725)
    ax.set_title("Precisión vs recall", color=PALETTE["navy"], fontweight="bold", pad=12)
    clean_axes(ax, grid_axis="both")
    savefig(fig, path)


# =============================================================================
# Modos
# =============================================================================

def run_inventory(project_root: Path, output_dir: Path, run_tags: Sequence[str]) -> pd.DataFrame:
    dirs = ensure_dirs(output_dir)
    inventory = discover_runs(project_root, run_tags)

    write_csv(inventory, dirs["Tables"] / "ResultsInventarioExperimentos.csv")

    lines = ["# Inventario de experimentos\n"]
    lines.append(f"- ProjectRoot: `{project_root}`")
    lines.append(f"- RunTags: `{', '.join(run_tags)}`")
    lines.append(f"- Experimentos detectados: `{len(inventory)}`\n")

    if inventory.empty:
        lines.append("No se detectaron experimentos válidos.")
    else:
        for _, row in inventory.iterrows():
            lines.append(
                f"- {row['ExperimentId']} · {row['SpanishModel']} · "
                f"{spanish_config(row['FeatureConfig'])} · RunTag `{row['RunTag']}` · "
                f"`{row['ModelRunId']}`"
            )
            lines.append(
                f"  - TrainingHistory: `{row['HasTrainingHistory']}`; "
                f"TestSummary: `{row['HasTestSummary']}`; "
                f"TestBySample: `{row['HasTestBySample']}`; "
                f"PredictionFigureIndex: `{row['HasPredictionFigureIndex']}`"
            )

    write_text("\n".join(lines) + "\n", dirs["Logs"] / "ResultsAuditoriaEvidencia.md")
    return inventory


def run_master(project_root: Path, output_dir: Path, run_tags: Sequence[str], metadata_csv: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dirs = ensure_dirs(output_dir)
    inventory = discover_runs(project_root, run_tags)
    summary = build_model_summary(inventory, run_tags)
    master = build_master_by_sample(inventory, run_tags)
    metadata = load_metadata(metadata_csv, project_root, run_tags)
    master = merge_metadata(master, metadata)

    if not summary.empty:
        write_csv(summary, dirs["Tables"] / "ResultsModelSummary.csv", digits=4)
        write_csv(summary, dirs["Latex"] / "ResultsModelSummaryForLatex.csv", digits=4)

    if not master.empty:
        write_csv(master, dirs["Tables"] / "ResultsMasterPerSample.csv", digits=4)
    else:
        write_text(
            "No existe una tabla completa por muestra para todos los experimentos detectados. "
            "Los análisis de fluxrate, tamaño de pluma, saturación y calidad binaria solo se generarán "
            "si TestMetricsBySample.csv contiene las columnas necesarias.\n",
            dirs["Logs"] / "ResultsMasterPerSampleNoDisponible.md",
        )

    return summary, master


def run_experiment(project_root: Path, output_dir: Path, run_tags: Sequence[str], experiment: str) -> None:
    dirs = ensure_dirs(output_dir)
    inventory = discover_runs(project_root, run_tags)
    summary = build_model_summary(inventory, run_tags)
    master = build_master_by_sample(inventory, run_tags)

    if experiment == "All":
        model_names = list(EXPERIMENTS.keys())
    else:
        model_names = [experiment]

    for model_name in model_names:
        build_experiment_outputs(model_name, inventory, summary, master, run_tags, dirs)


def run_factors(project_root: Path, output_dir: Path, run_tags: Sequence[str], metadata_csv: Optional[str] = None) -> None:
    dirs = ensure_dirs(output_dir)
    inventory = discover_runs(project_root, run_tags)
    master = build_master_by_sample(inventory, run_tags)
    metadata = load_metadata(metadata_csv, project_root, run_tags)
    master = merge_metadata(master, metadata)
    build_factors(master, dirs)



def run_fluxrate(project_root: Path, output_dir: Path, run_tags: Sequence[str], metadata_csv: Optional[str] = None) -> None:
    dirs = ensure_dirs(output_dir)
    inventory = discover_runs(project_root, run_tags)
    master = build_master_by_sample(inventory, run_tags)
    metadata = load_metadata(metadata_csv, project_root, run_tags)
    master = merge_metadata(master, metadata)
    build_fluxrate_outputs(master, dirs)


def run_thresholds(project_root: Path, output_dir: Path, run_tags: Sequence[str]) -> None:
    dirs = ensure_dirs(output_dir)
    inventory = discover_runs(project_root, run_tags)
    build_threshold_outputs(inventory, run_tags, dirs)


def run_summary(project_root: Path, output_dir: Path, run_tags: Sequence[str]) -> None:
    dirs = ensure_dirs(output_dir)
    inventory = discover_runs(project_root, run_tags)
    summary = build_model_summary(inventory, run_tags)
    build_summary_outputs(summary, dirs)


def build_manifest(output_dir: Path) -> None:
    rows = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            rows.append({
                "FileName": p.name,
                "RelativePath": str(p.relative_to(output_dir)),
                "FileType": p.suffix.replace(".", "").lower(),
            })
    write_csv(pd.DataFrame(rows), output_dir / "ResultsChapterManifest.csv")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera resultados del capítulo 5 por experimento.")
    parser.add_argument("--mode", choices=["inventory", "master", "experiment", "factors", "fluxrate", "thresholds", "summary", "all"], default="inventory")
    parser.add_argument("--ProjectRoot", type=str, default=".")
    parser.add_argument("--OutputDir", type=str, default="Outputs/ResultsChapter_101622_101840")
    parser.add_argument("--RunTags", type=str, default="101622,101840")
    parser.add_argument("--Experiment", choices=["SimpleUNet", "EnhancedUNet", "TransformerUNet", "TransformerPlus", "All"], default="All")
    parser.add_argument("--MetadataCsv", type=str, default=None, help="CSV opcional con SampleId, ch4_fluxrate y ch4_fluxrate_std para análisis físicos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()

    project_root = Path(args.ProjectRoot).expanduser().resolve()
    output_dir = Path(args.OutputDir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    run_tags = [x.strip() for x in args.RunTags.split(",") if x.strip()]
    ensure_dirs(output_dir)

    if args.mode == "inventory":
        run_inventory(project_root, output_dir, run_tags)

    elif args.mode == "master":
        run_master(project_root, output_dir, run_tags, args.MetadataCsv)

    elif args.mode == "experiment":
        run_experiment(project_root, output_dir, run_tags, args.Experiment)

    elif args.mode == "factors":
        run_factors(project_root, output_dir, run_tags, args.MetadataCsv)

    elif args.mode == "fluxrate":
        run_fluxrate(project_root, output_dir, run_tags, args.MetadataCsv)

    elif args.mode == "thresholds":
        run_thresholds(project_root, output_dir, run_tags)

    elif args.mode == "summary":
        run_summary(project_root, output_dir, run_tags)

    elif args.mode == "all":
        run_inventory(project_root, output_dir, run_tags)
        run_master(project_root, output_dir, run_tags, args.MetadataCsv)
        run_experiment(project_root, output_dir, run_tags, "All")
        run_thresholds(project_root, output_dir, run_tags)
        run_factors(project_root, output_dir, run_tags, args.MetadataCsv)
        run_summary(project_root, output_dir, run_tags)

    build_manifest(output_dir)
    print(f"[DONE] Resultados generados en: {output_dir}")


if __name__ == "__main__":
    main()