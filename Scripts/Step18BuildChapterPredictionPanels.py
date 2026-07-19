#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step18BuildChapterPredictionPanels.py

Genera paneles cualitativos específicos para el capítulo de resultados.

Objetivo:
- Reutilizar la lógica de Step13VisualizePredictions.py para generar imágenes desde cero.
- No depende de PNG previos.
- Carga tensores originales, checkpoints, reconstruye modelos y vuelve a calcular probabilidad/máscara.
- Produce paneles individuales 2x3 y comparaciones fijas 2x4 por SampleId.
- Exporta tablas CSV y LaTeX con métricas de cada caso.

Entradas esperadas:
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/<Split>Features.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/<Split>Masks.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Checkpoints/BestModel.pt
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Metrics/<Split>MetricsBySample.csv
- Outputs/Experiments/<RunTag>/<FeatureConfig>/<ModelName>_<RunName>/Tables/VisualizationCaseSet_<RunId>.csv

Salidas:
- Outputs/ResultsChapter_101622_101840/Figures/ChapterPredictionPanels/*.png
- Outputs/ResultsChapter_101622_101840/Tables/ChapterQualitativeCasesMetrics.csv
- Outputs/ResultsChapter_101622_101840/Tables/ChapterQualitativeCasesMetrics.tex
- Outputs/ResultsChapter_101622_101840/Tables/ChapterQualitativeFiguresIndex.csv

Uso recomendado:
python Scripts/Step18BuildChapterPredictionPanels.py \
  --ProjectRoot /data/users/kabasmen/MethaneProjectTFM \
  --OutputDir Outputs/ResultsChapter_101622_101840 \
  --RunTags 101622,101840 \
  --Split Test \
  --Device auto
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------
# Estilo gráfico común del proyecto
# ---------------------------------------------------------------------
COLORS = {
    "navy": "#0B2E6D",
    "blue": "#2B68C8",
    "lightblue": "#5AB4E5",
    "cyan": "#00D4FF",
    "green": "#22A65A",
    "yellow": "#F4E04D",
    "red": "#F51B23",
    "gray": "#E6EEF7",
    "linegray": "#AEBBCD",
    "text": "#102A56",
    "darkgray": "#6B7280",
    "white": "#FFFFFF",
    "black": "#000000",
}

MODEL_DISPLAY = {
    "SimpleUNet": "U-Net simple",
    "EnhancedUNet": "U-Net mejorada",
    "TransformerUNet": "U-Net Transformer",
    "TransformerPlus": "TransformerPlus",
}

CONFIG_DISPLAY = {
    "ConfigB": "Configuración B",
    "ConfigC": "Configuración C",
}

EXPERIMENT_DISPLAY = {
    "SimpleUNet": "Experimento 1",
    "EnhancedUNet": "Experimento 2",
    "TransformerUNet": "Experimento 3",
    "TransformerPlus": "Experimento 4",
}

DEFAULT_EXPERIMENTS = [
    {"ModelName": "SimpleUNet", "RunTag": "101622"},
    {"ModelName": "EnhancedUNet", "RunTag": "101622"},
    {"ModelName": "TransformerUNet", "RunTag": "101622"},
    {"ModelName": "TransformerPlus", "RunTag": "101840"},
]

METRIC_COLUMNS_CANONICAL = [
    "Threshold",
    "Dice",
    "IoU",
    "Precision",
    "Recall",
    "GTPixels",
    "PredPixels",
    "AreaRatio",
    "TP",
    "FP",
    "FN",
    "PredictedPositiveFraction",
]


@dataclass
class ModelSpec:
    model_name: str
    feature_config: str
    run_tag: str
    run_id: str
    run_name: str
    run_root: Path
    feature_dir: Path
    model_root: Path
    metrics_path: Path
    case_set_path: Path
    threshold: float


def pascal_case(value: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", str(value))
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def sanitize_filename(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def apply_chapter_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Montserrat",
            "figure.facecolor": COLORS["white"],
            "axes.facecolor": COLORS["white"],
            "axes.edgecolor": COLORS["linegray"],
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "savefig.facecolor": COLORS["white"],
        }
    )


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


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe CSV requerido: {path}")
    return pd.read_csv(path)


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


def find_model_root(run_root: Path, feature_config: str, model_name: str) -> tuple[Path, str, str]:
    config_root = run_root / feature_config
    if not config_root.exists():
        raise FileNotFoundError(f"No existe carpeta de configuración: {config_root}")

    candidates = []
    for p in config_root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(f"{model_name}_"):
            continue
        checkpoint = p / "Checkpoints" / "BestModel.pt"
        metrics = p / "Metrics" / "TestMetricsBySample.csv"
        score = int(checkpoint.exists()) + int(metrics.exists())
        candidates.append((score, p.stat().st_mtime, p))

    if not candidates:
        raise FileNotFoundError(
            f"No encontré modelo {model_name}_* en {config_root}. "
            "Verifica RunTag, FeatureConfig y ModelName."
        )

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    model_root = candidates[0][2]
    run_id = model_root.name
    run_name = run_id[len(model_name) + 1 :] if run_id.startswith(f"{model_name}_") else run_id
    return model_root, run_id, run_name


def find_threshold(model_root: Path, default: float = 0.5) -> float:
    candidates = [
        model_root / "Tables" / "BestThreshold.csv",
        model_root / "Metrics" / "BestThreshold.csv",
        model_root / "Tables" / "EvaluationSummary.csv",
        model_root / "Metrics" / "TestSummary.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for col in ["BestThreshold", "Threshold", "threshold", "best_threshold"]:
            if col in df.columns and len(df) > 0:
                val = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(val) > 0:
                    return float(val.iloc[0])

    training_config = model_root / "TrainingConfig.json"
    if training_config.exists():
        try:
            data = json.loads(training_config.read_text(encoding="utf-8"))
            for key in ["BestThreshold", "Threshold", "threshold"]:
                if key in data:
                    return float(data[key])
        except Exception:
            pass

    # Valor seguro para no romper; el usuario puede forzarlo con --ThresholdOverride.
    return float(default)


def build_model_specs(
    *,
    project_root: Path,
    run_tags: list[str],
    feature_configs: list[str],
    threshold_override: float | None,
) -> list[ModelSpec]:
    tag_set = set(run_tags)
    specs: list[ModelSpec] = []

    for exp in DEFAULT_EXPERIMENTS:
        model_name = exp["ModelName"]
        run_tag = exp["RunTag"]
        if run_tag not in tag_set:
            continue

        run_root = project_root / "Outputs" / "Experiments" / run_tag
        if not run_root.exists():
            print(f"[WARN] No existe RunRoot: {run_root}")
            continue

        for feature_config in feature_configs:
            try:
                model_root, run_id, run_name = find_model_root(run_root, feature_config, model_name)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}")
                continue

            threshold = float(threshold_override) if threshold_override is not None else find_threshold(model_root, default=0.5)
            specs.append(
                ModelSpec(
                    model_name=model_name,
                    feature_config=feature_config,
                    run_tag=run_tag,
                    run_id=run_id,
                    run_name=run_name,
                    run_root=run_root,
                    feature_dir=run_root / feature_config / "Features",
                    model_root=model_root,
                    metrics_path=model_root / "Metrics" / "TestMetricsBySample.csv",
                    case_set_path=model_root / "Tables" / f"VisualizationCaseSet_{run_id}.csv",
                    threshold=threshold,
                )
            )

    if not specs:
        raise RuntimeError("No se encontró ningún experimento válido.")

    return specs


def load_metrics_for_spec(spec: ModelSpec, split: str, step13) -> pd.DataFrame:
    metrics = read_csv_required(spec.model_root / "Metrics" / f"{split}MetricsBySample.csv")
    split_index = step13.load_split_index(spec.run_root, split, expected_n=len(metrics))

    if "SampleId" not in metrics.columns:
        if len(metrics) != len(split_index):
            raise ValueError(
                f"{spec.metrics_path} no tiene SampleId y len(metrics) != len(split_index)."
            )
        metrics["SampleId"] = split_index["SampleId"].values

    metrics = ensure_sample_id_column(metrics)
    metrics = metrics.merge(split_index[["SampleId", "SplitOrder"]], on="SampleId", how="left")
    return metrics


def get_metric_value(row: pd.Series, names: list[str], default: float = np.nan) -> float:
    for name in names:
        if name in row.index:
            try:
                val = float(row[name])
                return val
            except Exception:
                continue
    return float(default)


def canonical_metrics(row: pd.Series, threshold: float) -> dict[str, Any]:
    gt = get_metric_value(row, ["GroundTruthPixels", "GTPixels", "GT", "GTArea"], 0.0)
    pred = get_metric_value(row, ["PredictedPixels", "PredPixels", "PredArea"], 0.0)
    tp = get_metric_value(row, ["TruePositivePixels", "TP"], np.nan)
    fp = get_metric_value(row, ["FalsePositivePixels", "FP"], np.nan)
    fn = get_metric_value(row, ["FalseNegativePixels", "FN"], np.nan)

    area_ratio = get_metric_value(row, ["AreaRatio"], np.nan)
    if not np.isfinite(area_ratio) and gt > 0:
        area_ratio = pred / gt

    ppf = get_metric_value(row, ["PredictedPositiveFraction"], np.nan)
    if not np.isfinite(ppf):
        ppf = pred / 40000.0

    return {
        "Threshold": threshold,
        "Dice": get_metric_value(row, ["Dice"], np.nan),
        "IoU": get_metric_value(row, ["IoU"], np.nan),
        "Precision": get_metric_value(row, ["Precision"], np.nan),
        "Recall": get_metric_value(row, ["Recall"], np.nan),
        "GTPixels": int(round(gt)) if np.isfinite(gt) else "",
        "PredPixels": int(round(pred)) if np.isfinite(pred) else "",
        "AreaRatio": area_ratio,
        "TP": int(round(tp)) if np.isfinite(tp) else "",
        "FP": int(round(fp)) if np.isfinite(fp) else "",
        "FN": int(round(fn)) if np.isfinite(fn) else "",
        "PredictedPositiveFraction": ppf,
    }


def format_metric(value: Any, decimals: int = 3) -> str:
    if value == "" or value is None:
        return ""
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(x):
        return ""
    if abs(x - round(x)) < 1e-9 and abs(x) > 1:
        return str(int(round(x)))
    return f"{x:.{decimals}f}"


def load_raw_dataset(project_root: Path, step13, no_raw_context: bool) -> tuple[Any, pd.DataFrame | None]:
    if no_raw_context or not getattr(step13, "HAS_TACO_READERS", False):
        return None, None

    project_config_path = project_root / "Configs" / "ProjectConfig.yaml"
    if not project_config_path.exists():
        print(f"[WARN] No existe ProjectConfig: {project_config_path}. Se generarán paneles sin contexto raw.")
        return None, None

    try:
        project_config = step13.load_yaml(project_config_path)
        data_root, dataset_name = step13.resolve_dataset_location(project_config)
        if data_root is None:
            print("[WARN] ProjectConfig no define data_root/dataset. Se generarán paneles sin contexto raw.")
            return None, None
        return step13.load_dataset_flexible(data_root, dataset_name)
    except Exception as exc:
        print(f"[WARN] No pude cargar dataset raw: {repr(exc)}")
        return None, None


class PredictionEngine:
    def __init__(self, project_root: Path, spec: ModelSpec, split: str, device: Any, step13):
        self.project_root = project_root
        self.spec = spec
        self.split = split
        self.device = device
        self.step13 = step13

        self.feature_names = step13.load_feature_names(spec.feature_config)

        x_path = spec.feature_dir / f"{split}Features.npy"
        y_path = spec.feature_dir / f"{split}Masks.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"No existe X: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"No existe Y: {y_path}")

        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.split_index = step13.load_split_index(spec.run_root, split, expected_n=int(self.x.shape[0]))
        self.sample_to_index = {
            str(row["SampleId"]): int(row["SplitOrder"])
            for _, row in self.split_index.iterrows()
        }
        self.metrics = load_metrics_for_spec(spec, split, step13)
        self.metrics_by_sample = {str(r["SampleId"]): r for _, r in self.metrics.iterrows()}

        checkpoint_path = spec.model_root / "Checkpoints" / "BestModel.pt"
        checkpoint = step13.extract_checkpoint_payload(checkpoint_path, device=device)
        training_config = step13.load_json_if_exists(spec.model_root / "TrainingConfig.json")
        model_parameters = step13.get_model_parameters(checkpoint, training_config)

        input_channels = int(self.x.shape[1])
        checkpoint_input_channels = checkpoint.get("input_channels", checkpoint.get("InputChannels", None))
        if checkpoint_input_channels is not None and int(checkpoint_input_channels) != input_channels:
            raise ValueError(
                f"{spec.run_id}: checkpoint input_channels={checkpoint_input_channels}, "
                f"pero X tiene {input_channels}."
            )

        self.model = step13.build_model(
            model_name=spec.model_name,
            input_channels=input_channels,
            output_channels=1,
            model_parameters=model_parameters,
        )
        state_dict = step13.get_state_dict_from_checkpoint(checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device)
        self.model.eval()

    def get_case(self, sample_id: str, dataset: Any = None, sample_table: pd.DataFrame | None = None) -> dict[str, Any]:
        sample_id = str(sample_id)
        if sample_id not in self.sample_to_index:
            raise KeyError(f"SampleId {sample_id} no está en split {self.split} para {self.spec.run_id}.")

        idx = self.sample_to_index[sample_id]
        x_i = np.array(self.x[idx], dtype=np.float32, copy=True)
        y_i = np.array(self.y[idx], dtype=np.uint8, copy=True)
        prob, pred = self.step13.predict_one(
            model=self.model,
            x=x_i,
            device=self.device,
            threshold=float(self.spec.threshold),
        )
        row = self.metrics_by_sample.get(sample_id)
        if row is None:
            row = pd.Series({"SampleId": sample_id})

        raw_sample = None
        if dataset is not None:
            raw_sample = self.step13.read_sample_flexible(dataset, sample_table, sample_id)

        return {
            "SampleId": sample_id,
            "x": x_i,
            "y": y_i,
            "prob": prob,
            "pred": pred,
            "row": row,
            "sample": raw_sample,
            "feature_names": self.feature_names,
            "spec": self.spec,
        }


def show_target_or_reference(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    sample = case["sample"]
    x = case["x"]
    feature_names = case["feature_names"]

    target_cube = step13.get_sample_array(sample, "Target")
    reference_cube = step13.get_sample_array(sample, "Reference")
    cube = target_cube if target_cube is not None else reference_cube
    rgb, title = step13.make_swir_nir_blue_composite(cube, x, feature_names)
    ax.imshow(rgb)
    ax.set_title("Imagen target o reference", color=COLORS["text"], fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def show_ch4_or_mbmp(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    sample = case["sample"]
    x = case["x"]
    feature_names = case["feature_names"]
    gt = (case["y"][0] > 0).astype(np.uint8)

    ch4_arr = step13.get_sample_array(sample, "CH4")
    if ch4_arr is not None:
        ch4_chw = step13.to_chw(ch4_arr)
        if ch4_chw is not None:
            ch4_map = ch4_chw[0]
        else:
            ch4_map = np.asarray(ch4_arr)
            if ch4_map.ndim == 3:
                ch4_map = ch4_map[..., 0]
        ax.imshow(step13.normalize_for_display(ch4_map), cmap="GnBu", vmin=0, vmax=1)
        step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=2.0)
        ax.set_title("CH4", color=COLORS["text"], fontsize=11, fontweight="bold")
    else:
        mbmp, mbmp_name = step13.require_mbmp_base(x, feature_names)
        ax.imshow(step13.normalize_for_display(mbmp), cmap="Blues", vmin=0, vmax=1)
        step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=2.0)
        ax.set_title(mbmp_name, color=COLORS["text"], fontsize=11, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])


def show_ground_truth(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    gt = (case["y"][0] > 0).astype(np.uint8)
    ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=2.5)
    ax.set_title("Ground truth", color=COLORS["text"], fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def show_probability(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    gt = (case["y"][0] > 0).astype(np.uint8)
    pred = (case["pred"] > 0).astype(np.uint8)
    ax.imshow(case["prob"], cmap="viridis", vmin=0, vmax=1)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=2.0)
    step13.safe_contour(ax, pred, color=COLORS["cyan"], linewidth=2.0)
    ax.set_title("Probabilidad predicha", color=COLORS["text"], fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def show_predicted_mask(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    gt = (case["y"][0] > 0).astype(np.uint8)
    pred = (case["pred"] > 0).astype(np.uint8)
    canvas = np.zeros_like(pred, dtype=np.float32)
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(pred == 0, pred), cmap="winter", alpha=0.72)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=2.0)
    step13.safe_contour(ax, pred, color=COLORS["cyan"], linewidth=2.2)
    ax.set_title("Máscara predicha", color=COLORS["text"], fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def show_overlay(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    gt = (case["y"][0] > 0).astype(np.uint8)
    pred = (case["pred"] > 0).astype(np.uint8)
    ax.imshow(step13.make_overlay_rgb(gt, pred), vmin=0, vmax=1)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=1.2)
    step13.safe_contour(ax, pred, color=COLORS["cyan"], linewidth=1.2)
    ax.set_title("Overlay TP/FP/FN", color=COLORS["text"], fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def add_common_legend(fig: plt.Figure, y: float = 0.02, fontsize: int = 10) -> None:
    handles = [
        mpatches.Patch(color=COLORS["red"], label="Ground truth / FN"),
        mpatches.Patch(color=COLORS["cyan"], label="Predicción"),
        mpatches.Patch(color=COLORS["green"], label="TP"),
        mpatches.Patch(color=COLORS["yellow"], label="FP"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        fontsize=fontsize,
    )


def create_individual_panel(
    *,
    case: dict[str, Any],
    output_path: Path,
    case_group_label: str,
    step13,
) -> dict[str, Any]:
    spec: ModelSpec = case["spec"]
    row: pd.Series = case["row"]
    sample_id = case["SampleId"]

    exp_label = EXPERIMENT_DISPLAY.get(spec.model_name, spec.model_name)
    model_label = MODEL_DISPLAY.get(spec.model_name, spec.model_name)
    config_label = CONFIG_DISPLAY.get(spec.feature_config, spec.feature_config)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    axes = axes.ravel()

    show_target_or_reference(axes[0], case, step13)
    show_ch4_or_mbmp(axes[1], case, step13)
    show_ground_truth(axes[2], case, step13)
    show_probability(axes[3], case, step13)
    show_predicted_mask(axes[4], case, step13)
    show_overlay(axes[5], case, step13)

    title = (
        f"{exp_label}: {model_label} · {config_label}\n"
        f"{case_group_label} · SampleId={sample_id}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", color=COLORS["navy"], y=0.975)
    add_common_legend(fig, y=0.005, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.055, 0.98, 0.92])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics = canonical_metrics(row, spec.threshold)
    return {
        "FigurePath": str(output_path),
        "RunTag": spec.run_tag,
        "Experiment": exp_label,
        "ModelName": spec.model_name,
        "ModelLabel": model_label,
        "FeatureConfig": spec.feature_config,
        "ConfigLabel": config_label,
        "RunId": spec.run_id,
        "CaseGroup": case_group_label,
        "SampleId": sample_id,
        **metrics,
    }


def create_fixed_comparison_panel(
    *,
    sample_id: str,
    engines: dict[tuple[str, str], PredictionEngine],
    specs: list[ModelSpec],
    output_path: Path,
    dataset: Any,
    sample_table: pd.DataFrame | None,
    step13,
) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.2))
    config_order = ["ConfigB", "ConfigC"]
    model_order = ["SimpleUNet", "EnhancedUNet", "TransformerUNet", "TransformerPlus"]

    rows_out: list[dict[str, Any]] = []

    for r, feature_config in enumerate(config_order):
        for c, model_name in enumerate(model_order):
            ax = axes[r, c]
            key = (model_name, feature_config)
            engine = engines.get(key)

            if engine is None:
                ax.text(0.5, 0.5, "Modelo no disponible", ha="center", va="center", color=COLORS["darkgray"])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{MODEL_DISPLAY.get(model_name, model_name)}\n{CONFIG_DISPLAY[feature_config]}", fontsize=10, color=COLORS["text"], fontweight="bold")
                continue

            try:
                case = engine.get_case(sample_id, dataset=dataset, sample_table=sample_table)
                show_overlay(ax, case, step13)
                row = case["row"]
                metrics = canonical_metrics(row, engine.spec.threshold)
                dice = format_metric(metrics["Dice"])
                iou = format_metric(metrics["IoU"])
                ax.set_title(
                    f"{MODEL_DISPLAY.get(model_name, model_name)}\n{CONFIG_DISPLAY[feature_config]} · Dice={dice} · IoU={iou}",
                    fontsize=9,
                    color=COLORS["text"],
                    fontweight="bold",
                )
                rows_out.append(
                    {
                        "FigurePath": str(output_path),
                        "RunTag": engine.spec.run_tag,
                        "Experiment": EXPERIMENT_DISPLAY.get(model_name, model_name),
                        "ModelName": model_name,
                        "ModelLabel": MODEL_DISPLAY.get(model_name, model_name),
                        "FeatureConfig": feature_config,
                        "ConfigLabel": CONFIG_DISPLAY[feature_config],
                        "RunId": engine.spec.run_id,
                        "CaseGroup": "FixedComparisonCases",
                        "SampleId": sample_id,
                        **metrics,
                    }
                )
            except Exception as exc:
                ax.text(
                    0.5,
                    0.5,
                    f"No disponible\n{type(exc).__name__}",
                    ha="center",
                    va="center",
                    color=COLORS["darkgray"],
                    fontsize=9,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"{MODEL_DISPLAY.get(model_name, model_name)}\n{CONFIG_DISPLAY[feature_config]}",
                    fontsize=9,
                    color=COLORS["text"],
                    fontweight="bold",
                )

    fig.suptitle(
        f"Comparación fija por modelo y configuración\nSampleId={sample_id}",
        fontsize=16,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.975,
    )
    add_common_legend(fig, y=0.005, fontsize=10)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.90])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rows_out


def select_reference_fixed_sample_ids(
    *,
    specs: list[ModelSpec],
    fixed_count: int,
    reference_model: str,
    reference_config: str,
    split: str,
    step13,
) -> list[str]:
    ref = None
    for spec in specs:
        if spec.model_name == reference_model and spec.feature_config == reference_config:
            ref = spec
            break

    if ref is None:
        # Fallback: usa el primer TransformerPlus disponible o el último spec.
        for spec in specs:
            if spec.model_name == "TransformerPlus":
                ref = spec
                break
    if ref is None:
        ref = specs[-1]

    if ref.case_set_path.exists():
        cases = pd.read_csv(ref.case_set_path)
        cases = ensure_sample_id_column(cases)
        if "CaseGroup" in cases.columns:
            fixed = cases[cases["CaseGroup"].astype(str).eq("FixedComparisonCases")].copy()
            if len(fixed) > 0:
                if "Order" in fixed.columns:
                    fixed = fixed.sort_values("Order")
                return fixed["SampleId"].astype(str).head(fixed_count).tolist()

    metrics = load_metrics_for_spec(ref, split, step13)
    if "GroundTruthPixels" in metrics.columns:
        metrics = metrics[metrics["GroundTruthPixels"] > 0].copy()
    metrics = metrics.sort_values(["GroundTruthPixels", "SplitOrder"], ascending=[False, True])
    return metrics["SampleId"].astype(str).head(fixed_count).tolist()


def select_individual_cases(
    *,
    specs: list[ModelSpec],
    groups: list[str],
    per_group: int,
) -> list[tuple[ModelSpec, str, str]]:
    selected: list[tuple[ModelSpec, str, str]] = []

    # Para no saturar el capítulo, por defecto se priorizan EnhancedUNet y TransformerPlus.
    preferred_models = {"EnhancedUNet", "TransformerPlus"}

    for spec in specs:
        if spec.model_name not in preferred_models:
            continue
        if not spec.case_set_path.exists():
            print(f"[WARN] No existe case set: {spec.case_set_path}")
            continue
        cases = pd.read_csv(spec.case_set_path)
        cases = ensure_sample_id_column(cases)
        if "CaseGroup" not in cases.columns:
            continue
        for group in groups:
            sub = cases[cases["CaseGroup"].astype(str).eq(group)].copy()
            if "Order" in sub.columns:
                sub = sub.sort_values("Order")
            for _, row in sub.head(per_group).iterrows():
                selected.append((spec, str(row["SampleId"]), group))

    return selected


def dataframe_to_latex_table(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    keep = [
        "CaseGroup",
        "Experiment",
        "ModelLabel",
        "ConfigLabel",
        "SampleId",
        "Threshold",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "GTPixels",
        "PredPixels",
        "AreaRatio",
        "TP",
        "FP",
        "FN",
        "PredictedPositiveFraction",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    for col in ["Threshold", "Dice", "IoU", "Precision", "Recall", "AreaRatio", "PredictedPositiveFraction"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: format_metric(x, 3))

    rename = {
        "CaseGroup": "Caso",
        "Experiment": "Experimento",
        "ModelLabel": "Modelo",
        "ConfigLabel": "Configuración",
        "SampleId": "SampleId",
        "Threshold": "Umbral",
        "Dice": "Dice",
        "IoU": "IoU",
        "Precision": "Precisión",
        "Recall": "Recall",
        "GTPixels": "GT pixels",
        "PredPixels": "Pred. pixels",
        "AreaRatio": "Relación área",
        "TP": "TP",
        "FP": "FP",
        "FN": "FN",
        "PredictedPositiveFraction": "Fracción pred.",
    }
    out = out.rename(columns=rename)

    latex = out.to_latex(index=False, escape=True, longtable=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera paneles cualitativos para capítulo de resultados.")
    parser.add_argument("--ProjectRoot", default=".")
    parser.add_argument("--OutputDir", default="Outputs/ResultsChapter_101622_101840")
    parser.add_argument("--RunTags", default="101622,101840")
    parser.add_argument("--FeatureConfigs", default="ConfigB,ConfigC")
    parser.add_argument("--Split", default="Test", choices=["Train", "Validation", "Test"])
    parser.add_argument("--Device", default="auto")
    parser.add_argument("--ThresholdOverride", type=float, default=None)
    parser.add_argument("--FixedCount", type=int, default=4)
    parser.add_argument("--ReferenceModel", default="TransformerPlus")
    parser.add_argument("--ReferenceConfig", default="ConfigB", choices=["ConfigB", "ConfigC"])
    parser.add_argument("--IndividualGroups", default="BestPredictions,ErrorCases,WorstPredictions")
    parser.add_argument("--IndividualPerGroup", type=int, default=1)
    parser.add_argument("--NoRawContext", action="store_true")

    args = parser.parse_args()

    project_root = Path(args.ProjectRoot).resolve()
    output_dir = Path(args.OutputDir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    figures_dir = output_dir / "Figures" / "ChapterPredictionPanels"
    tables_dir = output_dir / "Tables"
    logs_dir = output_dir / "Logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    apply_chapter_style()
    step13 = import_step13(project_root)
    device = step13.get_device(args.Device)

    run_tags = [x.strip() for x in args.RunTags.split(",") if x.strip()]
    feature_configs = [x.strip() for x in args.FeatureConfigs.split(",") if x.strip()]
    individual_groups = [x.strip() for x in args.IndividualGroups.split(",") if x.strip()]

    specs = build_model_specs(
        project_root=project_root,
        run_tags=run_tags,
        feature_configs=feature_configs,
        threshold_override=args.ThresholdOverride,
    )

    print("\n=== STEP18 START ===")
    print(f"ProjectRoot: {project_root}")
    print(f"OutputDir: {output_dir}")
    print(f"Device: {device}")
    print("Experimentos encontrados:")
    for s in specs:
        print(f"- {s.run_tag} | {s.model_name} | {s.feature_config} | {s.run_id} | threshold={s.threshold:.3f}")

    dataset, sample_table = load_raw_dataset(project_root, step13, args.NoRawContext)
    print(f"RawContextAvailable: {dataset is not None}")

    engines: dict[tuple[str, str], PredictionEngine] = {}
    for spec in specs:
        key = (spec.model_name, spec.feature_config)
        try:
            engines[key] = PredictionEngine(project_root, spec, args.Split, device, step13)
        except Exception as exc:
            print(f"[WARN] No pude inicializar {key}: {repr(exc)}")

    if not engines:
        raise RuntimeError("No se pudo inicializar ningún motor de predicción.")

    all_metric_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []

    # 1) Comparaciones fijas 2x4.
    fixed_sample_ids = select_reference_fixed_sample_ids(
        specs=specs,
        fixed_count=args.FixedCount,
        reference_model=args.ReferenceModel,
        reference_config=args.ReferenceConfig,
        split=args.Split,
        step13=step13,
    )

    print("\nFixedComparison SampleIds:")
    for sid in fixed_sample_ids:
        print(f"- {sid}")

    for i, sample_id in enumerate(fixed_sample_ids, start=1):
        out_path = figures_dir / f"FigureChapterFixedComparisonCase{i:02d}.png"
        rows = create_fixed_comparison_panel(
            sample_id=sample_id,
            engines=engines,
            specs=specs,
            output_path=out_path,
            dataset=dataset,
            sample_table=sample_table,
            step13=step13,
        )
        all_metric_rows.extend(rows)
        figure_rows.append(
            {
                "FigureId": f"FigureChapterFixedComparisonCase{i:02d}",
                "FigurePath": str(out_path),
                "FigureType": "FixedComparisonCases",
                "SampleId": sample_id,
                "Description": "Comparación fija 2x4 por modelo y configuración",
            }
        )
        print(f"[FIG] {out_path}")

    # 2) Paneles individuales 2x3 para casos representativos.
    individual_specs = select_individual_cases(
        specs=specs,
        groups=individual_groups,
        per_group=args.IndividualPerGroup,
    )

    individual_counter = 0
    for spec, sample_id, group in individual_specs:
        key = (spec.model_name, spec.feature_config)
        engine = engines.get(key)
        if engine is None:
            continue
        try:
            case = engine.get_case(sample_id, dataset=dataset, sample_table=sample_table)
            individual_counter += 1
            group_label = {
                "BestPredictions": "Mejor predicción",
                "ErrorCases": "Caso de error",
                "WorstPredictions": "Predicción deficiente",
                "FixedComparisonCases": "Caso fijo comparativo",
            }.get(group, group)

            file_name = (
                f"FigureChapterIndividual{individual_counter:02d}"
                f"{pascal_case(group)}"
                f"{pascal_case(spec.model_name)}"
                f"{pascal_case(spec.feature_config)}.png"
            )
            out_path = figures_dir / file_name
            metric_row = create_individual_panel(
                case=case,
                output_path=out_path,
                case_group_label=group_label,
                step13=step13,
            )
            all_metric_rows.append(metric_row)
            figure_rows.append(
                {
                    "FigureId": Path(file_name).stem,
                    "FigurePath": str(out_path),
                    "FigureType": group,
                    "SampleId": sample_id,
                    "ModelName": spec.model_name,
                    "FeatureConfig": spec.feature_config,
                    "Description": f"Panel individual 2x3: {group_label}",
                }
            )
            print(f"[FIG] {out_path}")
        except Exception as exc:
            print(f"[WARN] No pude generar panel individual {spec.run_id} {sample_id}: {repr(exc)}")

    metrics_df = pd.DataFrame(all_metric_rows)
    figures_df = pd.DataFrame(figure_rows)

    metrics_path = tables_dir / "ChapterQualitativeCasesMetrics.csv"
    figures_path = tables_dir / "ChapterQualitativeFiguresIndex.csv"
    latex_path = tables_dir / "ChapterQualitativeCasesMetrics.tex"

    metrics_df.to_csv(metrics_path, index=False)
    figures_df.to_csv(figures_path, index=False)
    dataframe_to_latex_table(metrics_df, latex_path)

    run_config = {
        "ProjectRoot": str(project_root),
        "OutputDir": str(output_dir),
        "RunTags": run_tags,
        "FeatureConfigs": feature_configs,
        "Split": args.Split,
        "FixedCount": args.FixedCount,
        "ReferenceModel": args.ReferenceModel,
        "ReferenceConfig": args.ReferenceConfig,
        "IndividualGroups": individual_groups,
        "IndividualPerGroup": args.IndividualPerGroup,
        "RawContextAvailable": dataset is not None,
        "ModelSpecs": [s.__dict__ | {"run_root": str(s.run_root), "feature_dir": str(s.feature_dir), "model_root": str(s.model_root), "metrics_path": str(s.metrics_path), "case_set_path": str(s.case_set_path)} for s in specs],
    }
    (logs_dir / "Step18RunConfig.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== STEP18 COMPLETED ===")
    print(f"FiguresDir: {figures_dir}")
    print(f"MetricsCSV: {metrics_path}")
    print(f"FiguresIndexCSV: {figures_path}")
    print(f"LatexTable: {latex_path}")


if __name__ == "__main__":
    main()
