#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step18BuildChapterPredictionPanels.py

Genera paneles cualitativos específicos para el capítulo de resultados.

Objetivo:
- Reutilizar la lógica de Step13VisualizePredictions.py para generar imágenes desde cero.
- No depende de PNG previos.
- Carga tensores originales, checkpoints, reconstruye modelos y vuelve a calcular probabilidad/máscara.
- Produce paneles individuales 2x3 y comparaciones fijas 5x2 por SampleId.
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
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------
# Estilo gráfico común del proyecto
# ---------------------------------------------------------------------

STEP18_VERSION = "Step18_20260706_v7_AllModels_QualityFiltered"

COLORS = {
    "navy": "#0B2E6D",
    "blue": "#2B68C8",
    "lightblue": "#5AB4E5",
    "cyan": "#00D4FF",
    "green": "#22A65A",
    "yellow": "#F4E04D",
    "magenta": "#D946EF",
    "red": "#F51B23",
    "wind": "#269B7E",
    "plume_axis": "#8E44AD",
    "gray": "#E6EEF7",
    "linegray": "#AEBBCD",
    "lightgray": "#D7E0EC",
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


def _model_dir_matches(dir_name: str, requested_model_name: str) -> bool:
    """Resuelve modelos lógicos frente a nombres reales de carpetas.

    En este proyecto TransformerPlus fue entrenado como una variante de
    TransformerUNet; por eso la carpeta real empieza por TransformerUNet_
    y termina/contiene TransformerPlus.
    """
    if requested_model_name == "TransformerPlus":
        return dir_name.startswith("TransformerPlus_") or (
            dir_name.startswith("TransformerUNet_") and "TransformerPlus" in dir_name
        )

    if requested_model_name == "TransformerUNet":
        return dir_name.startswith("TransformerUNet_") and "TransformerPlus" not in dir_name

    return dir_name.startswith(f"{requested_model_name}_")


def _infer_run_name(dir_name: str, requested_model_name: str) -> str:
    if requested_model_name == "TransformerPlus" and dir_name.startswith("TransformerUNet_"):
        return dir_name[len("TransformerUNet_"):]

    if dir_name.startswith(f"{requested_model_name}_"):
        return dir_name[len(requested_model_name) + 1:]

    return dir_name.split("_", 1)[1] if "_" in dir_name else dir_name


def _model_name_for_build(requested_model_name: str) -> str:
    """Nombre que entiende CreateModel. TransformerPlus usa clase TransformerUNet."""
    if requested_model_name == "TransformerPlus":
        return "TransformerUNet"
    return requested_model_name


def find_model_root(run_root: Path, feature_config: str, model_name: str) -> tuple[Path, str, str]:
    config_root = run_root / feature_config
    if not config_root.exists():
        raise FileNotFoundError(f"No existe carpeta de configuración: {config_root}")

    candidates = []
    for p in config_root.iterdir():
        if not p.is_dir():
            continue

        if not _model_dir_matches(p.name, model_name):
            continue

        checkpoint = p / "Checkpoints" / "BestModel.pt"
        metrics = p / "Metrics" / "TestMetricsBySample.csv"
        case_set = p / "Tables" / f"VisualizationCaseSet_{p.name}.csv"

        score = int(checkpoint.exists()) + int(metrics.exists()) + int(case_set.exists())

        # Prioriza explícitamente el experimento potenciado cuando el modelo lógico es TransformerPlus.
        if model_name == "TransformerPlus" and "TransformerPlus" in p.name:
            score += 10

        # Evita que TransformerUNet base capture por accidente la carpeta TransformerPlus.
        if model_name == "TransformerUNet" and "TransformerPlus" not in p.name:
            score += 2

        candidates.append((score, p.stat().st_mtime, p))

    if not candidates:
        existing = ", ".join([q.name for q in config_root.iterdir() if q.is_dir()])
        raise FileNotFoundError(
            f"No encontré carpeta compatible con el modelo lógico {model_name} en {config_root}. "
            f"Carpetas disponibles: {existing}"
        )

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    model_root = candidates[0][2]
    run_id = model_root.name
    run_name = _infer_run_name(run_id, model_name)
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



def _apply_panel_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(COLORS["linegray"])
        spine.set_linewidth(0.9)


def add_minmax_colorbar(ax: plt.Axes, image_artist, vmin: float = 0.0, vmax: float = 1.0, tick_fontsize: int = 8):
    """Barra de color compacta con sólo mínimo y máximo, sin título."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = plt.colorbar(image_artist, cax=cax)
    cb.set_ticks([vmin, vmax])
    cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    cb.ax.tick_params(labelsize=tick_fontsize, colors=COLORS["text"], length=2)
    cb.outline.set_edgecolor(COLORS["linegray"])
    cb.outline.set_linewidth(0.8)
    return cb


def _pick_first_non_empty(row: pd.Series | None, candidates: list[str]) -> str | None:
    if row is None:
        return None
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                continue
            sval = str(val).strip()
            if sval and sval.lower() not in {"nan", "none", "unknown"}:
                return sval
    return None


def _normalize_sample_table(sample_table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Devuelve una copia de la tabla con columna SampleId cuando sea posible."""
    if sample_table is None:
        return None
    table = sample_table.copy()
    if "SampleId" in table.columns:
        table["SampleId"] = table["SampleId"].astype(str)
        return table
    for candidate in [
        "sample_id", "sampleId", "SampleID", "id", "Id", "ID",
        "uuid", "UUID", "image_id", "ImageId", "ImageID", "plume_id", "PlumeId"
    ]:
        if candidate in table.columns:
            table["SampleId"] = table[candidate].astype(str)
            return table
    return table


def _stringify_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    sval = str(value).strip()
    if not sval or sval.lower() in {"nan", "none", "unknown", "null", "[]", "{}"}:
        return None
    return sval


def _flatten_metadata(obj: Any, prefix: str = "", max_depth: int = 4) -> dict[str, Any]:
    """Aplana metadatos de dict/objeto para buscar país, fecha y lugar de forma robusta."""
    out: dict[str, Any] = {}
    if obj is None or max_depth < 0:
        return out

    if isinstance(obj, pd.Series):
        obj = obj.to_dict()

    if isinstance(obj, dict):
        items = obj.items()
    else:
        items = []
        for name in [
            "metadata", "meta", "properties", "attrs", "data", "record", "sample",
            "country", "Country", "date", "Date", "location", "Location",
            "site", "Site", "facility", "Facility", "region", "Region"
        ]:
            if hasattr(obj, name):
                try:
                    items = list(items) + [(name, getattr(obj, name))]
                except Exception:
                    pass

    for key, value in items:
        skey = str(key)
        full_key = f"{prefix}.{skey}" if prefix else skey
        if isinstance(value, dict) or isinstance(value, pd.Series):
            out.update(_flatten_metadata(value, full_key, max_depth - 1))
        else:
            # Evita expandir arrays grandes de imagen.
            if isinstance(value, (list, tuple)) and len(value) > 0 and len(value) <= 8:
                out[full_key] = ", ".join(str(v) for v in value)
            elif not hasattr(value, "shape"):
                out[full_key] = value
    return out


def _pick_from_flat(flat: dict[str, Any], candidates: list[str]) -> str | None:
    if not flat:
        return None
    lower_map = {str(k).lower(): k for k in flat.keys()}
    # Primero coincidencia exacta por nombre final de clave.
    for candidate in candidates:
        cand = candidate.lower()
        for lk, original_key in lower_map.items():
            last = lk.split(".")[-1]
            if last == cand:
                val = _stringify_metadata_value(flat[original_key])
                if val is not None:
                    return val
    # Luego coincidencia parcial conservadora.
    for candidate in candidates:
        cand = candidate.lower()
        for lk, original_key in lower_map.items():
            if cand in lk:
                val = _stringify_metadata_value(flat[original_key])
                if val is not None:
                    return val
    return None


def _format_date_value(date_raw: str | None) -> str:
    if date_raw is None:
        return "Unknown"
    try:
        return pd.to_datetime(date_raw).strftime("%Y-%m-%d")
    except Exception:
        return str(date_raw)


def extract_sample_metadata(sample_table: pd.DataFrame | None, sample_id: str) -> dict[str, str]:
    """Extrae país, fecha y lugar desde sample_table aunque la columna ID tenga otro nombre."""
    default = {"Country": "Unknown", "Date": "Unknown", "Location": "Unknown"}
    table = _normalize_sample_table(sample_table)
    if table is None or "SampleId" not in table.columns:
        return default

    sub = table[table["SampleId"].astype(str) == str(sample_id)]
    if len(sub) == 0:
        return default

    row = sub.iloc[0]
    flat = _flatten_metadata(row)

    country = _pick_from_flat(flat, [
        "Country", "country", "country_name", "CountryName", "PlumeCountry", "SceneCountry",
        "TargetCountry", "ReferenceCountry", "ISO3Country", "CountryCode", "iso3", "iso_code",
        "admin_country", "adm0_name", "ADM0_NAME", "nation", "Nation"
    ]) or "Unknown"

    date_raw = _pick_from_flat(flat, [
        "Date", "date", "AcquisitionDate", "CaptureDate", "UTCDate", "Datetime", "DateTime",
        "Timestamp", "SceneDate", "TargetDate", "SensingDate", "acquisition_time", "datetime"
    ])

    location = _pick_from_flat(flat, [
        "Location", "location", "Facility", "FacilityName", "AssetName", "Region", "Province",
        "State", "Basin", "SiteName", "SourceName", "TargetName", "ReferenceName", "SceneName",
        "site", "site_name", "plume_name", "source", "name"
    ]) or "Unknown"

    return {"Country": country, "Date": _format_date_value(date_raw), "Location": location}


def extract_sample_metadata_from_raw_sample(sample: Any) -> dict[str, str]:
    """Fallback: intenta leer país/fecha/lugar desde el objeto raw del dataset."""
    flat = _flatten_metadata(sample)
    country = _pick_from_flat(flat, [
        "Country", "country", "country_name", "CountryName", "PlumeCountry", "SceneCountry",
        "TargetCountry", "ReferenceCountry", "ISO3Country", "CountryCode", "iso3", "adm0_name", "ADM0_NAME"
    ]) or "Unknown"
    date_raw = _pick_from_flat(flat, [
        "Date", "date", "AcquisitionDate", "CaptureDate", "UTCDate", "Datetime", "DateTime",
        "Timestamp", "SceneDate", "TargetDate", "SensingDate", "acquisition_time", "datetime"
    ])
    location = _pick_from_flat(flat, [
        "Location", "location", "Facility", "FacilityName", "AssetName", "Region", "Province",
        "State", "Basin", "SiteName", "SourceName", "TargetName", "ReferenceName", "SceneName",
        "site", "site_name", "plume_name", "source", "name"
    ]) or "Unknown"
    return {"Country": country, "Date": _format_date_value(date_raw), "Location": location}


def merge_metadata(primary: dict[str, str], fallback: dict[str, str]) -> dict[str, str]:
    out = dict(primary or {})
    for key in ["Country", "Date", "Location"]:
        val = out.get(key, "Unknown")
        if val in {None, "", "Unknown"}:
            out[key] = fallback.get(key, "Unknown")
    return out


def describe_case_metadata(meta: dict[str, str]) -> str:
    return (
        f"País: {meta.get('Country', 'Unknown')} · "
        f"Fecha: {meta.get('Date', 'Unknown')} · "
        f"Lugar: {meta.get('Location', 'Unknown')}"
    )


def draw_orientation_axes(ax: plt.Axes) -> None:
    """Mini eje N/E para orientar visualmente el parche."""
    x0, y0 = 0.075, 0.12
    dx, dy = 0.075, 0.075
    ax.annotate("", xy=(x0 + dx, y0), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=COLORS["navy"]))
    ax.annotate("", xy=(x0, y0 + dy), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=COLORS["navy"]))
    ax.text(x0 + dx + 0.012, y0 - 0.006, "E", transform=ax.transAxes, fontsize=8,
            fontweight="bold", color=COLORS["navy"])
    ax.text(x0 - 0.008, y0 + dy + 0.012, "N", transform=ax.transAxes, fontsize=8,
            fontweight="bold", color=COLORS["navy"])


def compute_plume_principal_axis(mask: np.ndarray):
    """Calcula eje principal de pluma mediante PCA sobre píxeles positivos."""
    binary = np.asarray(mask) > 0
    ys, xs = np.nonzero(binary)
    if len(xs) < 5:
        return None

    cx = float(xs.mean())
    cy = float(ys.mean())
    coords = np.column_stack((xs - cx, ys - cy))
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = int(np.argmax(eigvals))
    direction = eigvecs[:, idx]
    dx, dy = float(direction[0]), float(direction[1])
    norm = float(np.hypot(dx, dy))
    if norm == 0:
        return None
    dx, dy = dx / norm, dy / norm
    length = float(np.clip(np.sqrt(max(float(eigvals[idx]), 1.0)) * 3.0, 18.0, 55.0))
    return cx, cy, dx, dy, length


def compute_alignment_angle_deg(plume_dx: float, plume_dy: float, wind_cos: float, wind_sin: float):
    """Ángulo mínimo entre eje axial de la pluma y dirección del viento."""
    if wind_cos is None or wind_sin is None:
        return None
    if not np.isfinite(wind_cos) or not np.isfinite(wind_sin):
        return None
    wind_norm = float(np.hypot(wind_cos, wind_sin))
    if wind_norm == 0:
        return None
    wx, wy = float(wind_cos) / wind_norm, float(wind_sin) / wind_norm
    # En imagen, y crece hacia abajo; por eso convertimos el eje visual a coordenadas cartesianas.
    pdx, pdy = plume_dx, -plume_dy
    pnorm = float(np.hypot(pdx, pdy))
    if pnorm == 0:
        return None
    pdx, pdy = pdx / pnorm, pdy / pnorm
    dot = abs(pdx * wx + pdy * wy)
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def draw_config_c_alignment(ax: plt.Axes, case: dict[str, Any], mask: np.ndarray, step13, show_text: bool = True) -> None:
    """Dibuja eje principal de pluma y vector de viento sólo para ConfigC."""
    spec: ModelSpec = case["spec"]
    if spec.feature_config != "ConfigC":
        return

    gt = np.asarray(mask) > 0
    axis = compute_plume_principal_axis(gt)
    if axis is None:
        return

    wind_speed, wind_cos, wind_sin = step13.get_wind_values(case["x"], case["feature_names"])

    cx, cy, dx, dy, length = axis
    ax.plot(
        [cx - dx * length, cx + dx * length],
        [cy - dy * length, cy + dy * length],
        color=COLORS["plume_axis"],
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
        zorder=20,
    )
    ax.scatter([cx], [cy], s=18, color=COLORS["plume_axis"], zorder=21)

    step13.draw_wind_arrow(ax, mask=gt.astype(np.uint8), wind_cos=wind_cos, wind_sin=wind_sin, color=COLORS["wind"])

    if show_text:
        angle = compute_alignment_angle_deg(dx, dy, wind_cos, wind_sin)
        if angle is not None:
            ax.text(
                0.03, 0.045, f"Angulo viento-pluma = {angle:.1f} deg",
                transform=ax.transAxes,
                fontsize=8,
                color=COLORS["text"],
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=COLORS["linegray"], alpha=0.92),
                zorder=30,
            )


def make_overlay_rgba(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Overlay sin naranja: TP verde, FP fucsia, FN amarillo sobre fondo blanco."""
    gt = np.asarray(gt) > 0
    pred = np.asarray(pred) > 0
    h, w = gt.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., 3] = 1.0

    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)

    rgba[tp] = np.array([0.13, 0.65, 0.35, 0.58], dtype=np.float32)  # verde
    rgba[fp] = np.array([0.85, 0.26, 0.90, 0.60], dtype=np.float32)  # fucsia
    rgba[fn] = np.array([0.98, 0.84, 0.18, 0.62], dtype=np.float32)  # amarillo
    return rgba


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

        build_model_name = _model_name_for_build(spec.model_name)
        self.model = step13.build_model(
            model_name=build_model_name,
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

        table_meta = extract_sample_metadata(sample_table, sample_id)
        raw_meta = extract_sample_metadata_from_raw_sample(raw_sample) if raw_sample is not None else {}
        meta = merge_metadata(table_meta, raw_meta)

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
            "meta": meta,
        }


def show_target_or_reference(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    sample = case["sample"]
    x = case["x"]
    feature_names = case["feature_names"]
    gt = (case["y"][0] > 0).astype(np.uint8)

    target_cube = step13.get_sample_array(sample, "Target")
    reference_cube = step13.get_sample_array(sample, "Reference")
    if target_cube is not None:
        cube = target_cube
        source_label = "Target"
    else:
        cube = reference_cube
        source_label = "Reference"

    rgb, combo_title = step13.make_swir_nir_blue_composite(cube, x, feature_names)
    ax.imshow(rgb)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=1.6)
    draw_orientation_axes(ax)
    ax.set_title(f"{source_label}\n{combo_title}", color=COLORS["text"], fontsize=10.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _apply_panel_frame(ax)


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
        img = step13.normalize_for_display(ch4_map)
        artist = ax.imshow(img, cmap="GnBu", vmin=0, vmax=1)
        title = "CH4"
    else:
        mbmp, mbmp_name = step13.require_mbmp_base(x, feature_names)
        img = step13.normalize_for_display(mbmp)
        artist = ax.imshow(img, cmap="GnBu", vmin=0, vmax=1)
        title = mbmp_name

    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=1.8)
    draw_config_c_alignment(ax, case, gt, step13, show_text=True)
    draw_orientation_axes(ax)
    ax.set_title(title, color=COLORS["text"], fontsize=10.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _apply_panel_frame(ax)
    add_minmax_colorbar(ax, artist, 0.0, 1.0)


def show_ground_truth(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    gt = (case["y"][0] > 0).astype(np.uint8)
    ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=2.4)
    draw_orientation_axes(ax)
    ax.set_title("Ground truth", color=COLORS["text"], fontsize=10.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _apply_panel_frame(ax)


def show_probability(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    gt = (case["y"][0] > 0).astype(np.uint8)
    pred = (case["pred"] > 0).astype(np.uint8)
    artist = ax.imshow(case["prob"], cmap="viridis", vmin=0, vmax=1)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=1.7)
    step13.safe_contour(ax, pred, color=COLORS["cyan"], linewidth=1.7)
    draw_config_c_alignment(ax, case, gt, step13, show_text=True)
    draw_orientation_axes(ax)
    ax.set_title("Mapa de probabilidad", color=COLORS["text"], fontsize=10.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _apply_panel_frame(ax)
    add_minmax_colorbar(ax, artist, 0.0, 1.0)


def show_predicted_mask(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    gt = (case["y"][0] > 0).astype(np.uint8)
    pred = (case["pred"] > 0).astype(np.uint8)

    ax.imshow(np.ones_like(pred, dtype=np.float32), cmap="gray", vmin=0, vmax=1)
    cyan_cmap = ListedColormap([COLORS["cyan"]])
    ax.imshow(np.ma.masked_where(pred == 0, pred), cmap=cyan_cmap, alpha=0.18, vmin=0, vmax=1)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=1.9)
    step13.safe_contour(ax, pred, color=COLORS["cyan"], linewidth=2.1)
    draw_config_c_alignment(ax, case, gt, step13, show_text=True)
    draw_orientation_axes(ax)
    ax.set_title("Máscara predicha", color=COLORS["text"], fontsize=10.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _apply_panel_frame(ax)


def show_overlay(ax: plt.Axes, case: dict[str, Any], step13) -> None:
    gt = (case["y"][0] > 0).astype(np.uint8)
    pred = (case["pred"] > 0).astype(np.uint8)
    ax.imshow(make_overlay_rgba(gt, pred), vmin=0, vmax=1)
    step13.safe_contour(ax, gt, color=COLORS["red"], linewidth=1.5)
    step13.safe_contour(ax, pred, color=COLORS["cyan"], linewidth=1.5)
    draw_config_c_alignment(ax, case, gt, step13, show_text=True)
    draw_orientation_axes(ax)
    ax.set_title("Overlay TP / FP / FN", color=COLORS["text"], fontsize=10.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    _apply_panel_frame(ax)


def add_common_legend(fig: plt.Figure, y: float = 0.02, fontsize: int = 9) -> None:
    handles = [
        Line2D([0], [0], color=COLORS["red"], lw=2.2, label="Ground truth"),
        Line2D([0], [0], color=COLORS["cyan"], lw=2.2, label="Predicción"),
        mpatches.Patch(facecolor=COLORS["green"], edgecolor="none", label="TP"),
        mpatches.Patch(facecolor=COLORS["magenta"], edgecolor="none", label="FP"),
        mpatches.Patch(facecolor=COLORS["yellow"], edgecolor="none", label="FN"),
        Line2D([0], [0], color=COLORS["wind"], lw=2.2, label="Viento"),
        Line2D([0], [0], color=COLORS["plume_axis"], lw=2.0, linestyle="--", label="Eje pluma"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        bbox_to_anchor=(0.5, y),
        fontsize=fontsize,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(COLORS["linegray"])
    legend.get_frame().set_alpha(0.96)


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
    meta = case.get("meta", {})

    exp_label = EXPERIMENT_DISPLAY.get(spec.model_name, spec.model_name)
    model_label = MODEL_DISPLAY.get(spec.model_name, spec.model_name)
    config_label = CONFIG_DISPLAY.get(spec.feature_config, spec.feature_config)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.8))
    axes = axes.ravel()

    show_target_or_reference(axes[0], case, step13)
    show_ch4_or_mbmp(axes[1], case, step13)
    show_ground_truth(axes[2], case, step13)
    show_probability(axes[3], case, step13)
    show_predicted_mask(axes[4], case, step13)
    show_overlay(axes[5], case, step13)

    title = f"{exp_label}: {model_label} · {config_label}\n{case_group_label}"
    fig.suptitle(
        title,
        fontsize=15.2,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.982,
        linespacing=1.35,
    )
    fig.text(0.5, 0.905, f"SampleId: {sample_id}", ha="center", va="center", fontsize=9.6, color=COLORS["text"])
    fig.text(0.5, 0.878, describe_case_metadata(meta), ha="center", va="center", fontsize=9.6, color=COLORS["text"])
    add_common_legend(fig, y=0.026, fontsize=8.4)
    fig.subplots_adjust(left=0.035, right=0.975, bottom=0.105, top=0.805, wspace=0.23, hspace=0.32)

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
        "Country": meta.get("Country", "Unknown"),
        "Date": meta.get("Date", "Unknown"),
        "Location": meta.get("Location", "Unknown"),
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
    fig, axes = plt.subplots(2, 5, figsize=(18.8, 9.6))
    config_order = ["ConfigB", "ConfigC"]
    model_order = ["SimpleUNet", "EnhancedUNet", "TransformerUNet", "TransformerPlus"]

    rows_out: list[dict[str, Any]] = []
    meta = extract_sample_metadata(sample_table, sample_id)

    # Columna 1: contexto común del caso.
    ref_case = None
    for preferred_key in [("TransformerPlus", "ConfigB"), ("EnhancedUNet", "ConfigB")]:
        engine = engines.get(preferred_key)
        if engine is not None:
            try:
                ref_case = engine.get_case(sample_id, dataset=dataset, sample_table=sample_table)
                break
            except Exception:
                pass

    if ref_case is not None:
        meta = merge_metadata(meta, ref_case.get("meta", {}))
        show_target_or_reference(axes[0, 0], ref_case, step13)
        axes[0, 0].set_title("Contexto RGB", fontsize=9.2, color=COLORS["text"], fontweight="bold")
        show_ch4_or_mbmp(axes[1, 0], ref_case, step13)
        axes[1, 0].set_title("Contexto CH4 / MBMP+", fontsize=9.2, color=COLORS["text"], fontweight="bold")
    else:
        for ax, title in [(axes[0, 0], "Contexto RGB"), (axes[1, 0], "Contexto CH4 / MBMP+")]:
            ax.text(0.5, 0.5, "Contexto no disponible", ha="center", va="center", color=COLORS["darkgray"], fontsize=9)
            ax.set_title(title, fontsize=9.2, color=COLORS["text"], fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            _apply_panel_frame(ax)

    for r, feature_config in enumerate(config_order):
        for c, model_name in enumerate(model_order, start=1):
            ax = axes[r, c]
            key = (model_name, feature_config)
            engine = engines.get(key)

            if engine is None:
                ax.text(0.5, 0.5, "Modelo no disponible", ha="center", va="center", color=COLORS["darkgray"])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"{MODEL_DISPLAY.get(model_name, model_name)}\n{CONFIG_DISPLAY[feature_config]}",
                    fontsize=8.5,
                    color=COLORS["text"],
                    fontweight="bold",
                )
                _apply_panel_frame(ax)
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
                    fontsize=8.2,
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
                        "Country": meta.get("Country", "Unknown"),
                        "Date": meta.get("Date", "Unknown"),
                        "Location": meta.get("Location", "Unknown"),
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
                    fontsize=8.5,
                    color=COLORS["text"],
                    fontweight="bold",
                )
                _apply_panel_frame(ax)

    fig.suptitle(
        "Comparación fija por modelo y configuración",
        fontsize=15.5,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.982,
    )
    fig.text(0.5, 0.942, f"SampleId: {sample_id}", ha="center", va="center", fontsize=9.4, color=COLORS["text"])
    fig.text(0.5, 0.916, describe_case_metadata(meta), ha="center", va="center", fontsize=9.4, color=COLORS["text"])
    add_common_legend(fig, y=0.026, fontsize=8.3)
    fig.subplots_adjust(left=0.018, right=0.982, bottom=0.105, top=0.815, wspace=0.24, hspace=0.42)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rows_out



def _bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)
    return df[column].fillna(default).astype(bool)


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] No pude leer {path}: {repr(exc)}")
    return None


def build_exclusion_registry(
    *,
    output_dir: Path,
    split: str,
    project_root: Path,
    exclude_quality_issues: bool,
    exclude_saturated: bool,
    quality_audit_path: Path | None,
    anomaly_audit_path: Path | None,
) -> tuple[set[str], pd.DataFrame, dict[str, Any]]:
    """Construye lista de SampleId que NO deben usarse en paneles cualitativos.

    Importante: esto no cambia métricas globales. Solo filtra selección visual.
    Criterios de calidad:
    - Target sospechoso o RGB inválido.
    - Target/Reference no disponible o totalmente cero.
    - Tensor completo vacío.
    - Productos derivados principales vacíos: B8A, B11, B12, NDSWIR, ratios, MBMP, MBMPPlus, DualEnhancement.
    Criterio de saturación:
    - IsSaturatedPrediction=True en Step19MetricsAnomalyAudit.
    """
    rows: list[dict[str, Any]] = []
    excluded: set[str] = set()

    default_quality = output_dir / "Tables" / "Step19SampleQualityAudit.csv"
    default_anomaly = output_dir / "Tables" / "Step19MetricsAnomalyAudit.csv"

    # Si se pasa ruta relativa, resolver contra ProjectRoot; si no existe, probar contra OutputDir.
    def resolve_path(path: Path | None, default_path: Path) -> Path:
        if path is None:
            return default_path
        if path.is_absolute():
            return path
        p1 = project_root / path
        if p1.exists():
            return p1
        p2 = output_dir / path
        if p2.exists():
            return p2
        return p1

    q_path = resolve_path(quality_audit_path, default_quality)
    a_path = resolve_path(anomaly_audit_path, default_anomaly)

    if exclude_quality_issues:
        q = _load_optional_csv(q_path)
        if q is None:
            print(f"[WARN] No encontré auditoría de calidad: {q_path}. No se filtrarán Target/Reference/productos.")
        else:
            q = ensure_sample_id_column(q)
            if "Split" in q.columns:
                q_use = q[q["Split"].astype(str).eq(split)].copy()
            else:
                q_use = q.copy()

            derived_cols = [
                "B8A_AllZero", "B11_AllZero", "B12_AllZero",
                "NDSWIR_AllZero", "RatioB12B11_AllZero", "RatioB12B8A_AllZero",
                "MBMP_AllZero", "MBMPPlus_AllZero", "DualEnhancementB12B11_AllZero",
            ]

            flags = pd.DataFrame(index=q_use.index)
            flags["RGBBlackOrInvalid"] = _bool_series(q_use, "RGBBlackOrInvalid")
            flags["TargetSuspect"] = _bool_series(q_use, "TargetSuspect")
            flags["TargetMissing"] = ~_bool_series(q_use, "TargetAvailable", default=True)
            flags["ReferenceMissing"] = ~_bool_series(q_use, "ReferenceAvailable", default=True)
            flags["TargetAllZero"] = _bool_series(q_use, "TargetAllZero")
            flags["ReferenceAllZero"] = _bool_series(q_use, "ReferenceAllZero")
            flags["FeatureAllZero"] = _bool_series(q_use, "FeatureAllZero")
            flags["DerivedProductAllZero"] = False
            for col in derived_cols:
                if col in q_use.columns:
                    flags["DerivedProductAllZero"] = flags["DerivedProductAllZero"] | _bool_series(q_use, col)

            bad_mask = flags.any(axis=1)
            bad = q_use.loc[bad_mask, [c for c in ["Split", "SampleId", "RunTag", "FeatureConfig", "Country", "Date", "Location"] if c in q_use.columns]].copy()
            if len(bad) > 0:
                # Colapsar por SampleId y guardar razones únicas.
                for sid, group_idx in q_use.loc[bad_mask].groupby("SampleId").groups.items():
                    idxs = list(group_idx)
                    reason_names = []
                    for reason_col in flags.columns:
                        if bool(flags.loc[idxs, reason_col].any()):
                            reason_names.append(reason_col)
                    first = q_use.loc[idxs[0]]
                    excluded.add(str(sid))
                    rows.append({
                        "Split": str(first.get("Split", split)),
                        "SampleId": str(sid),
                        "ReasonGroup": "QualityIssue",
                        "Reasons": ";".join(reason_names),
                        "Country": first.get("Country", ""),
                        "Date": first.get("Date", ""),
                        "Location": first.get("Location", ""),
                    })

    if exclude_saturated:
        a = _load_optional_csv(a_path)
        if a is None:
            print(f"[WARN] No encontré auditoría de anomalías: {a_path}. No se filtrarán saturados.")
        else:
            a = ensure_sample_id_column(a)
            if "Split" in a.columns:
                a_use = a[a["Split"].astype(str).eq(split)].copy()
            else:
                a_use = a.copy()
            sat = a_use[_bool_series(a_use, "IsSaturatedPrediction")].copy()
            for sid, group in sat.groupby("SampleId"):
                excluded.add(str(sid))
                models = sorted(set(group.get("ModelName", pd.Series(dtype=str)).astype(str).tolist())) if "ModelName" in group.columns else []
                configs = sorted(set(group.get("FeatureConfig", pd.Series(dtype=str)).astype(str).tolist())) if "FeatureConfig" in group.columns else []
                rows.append({
                    "Split": str(group.iloc[0].get("Split", split)),
                    "SampleId": str(sid),
                    "ReasonGroup": "SaturatedPrediction",
                    "Reasons": f"IsSaturatedPrediction;Models={','.join(models)};Configs={','.join(configs)}",
                    "Country": "",
                    "Date": "",
                    "Location": "",
                })

    excl_df = pd.DataFrame(rows)
    if len(excl_df) > 0:
        excl_df = excl_df.drop_duplicates(["Split", "SampleId", "ReasonGroup", "Reasons"]).sort_values(["Split", "SampleId", "ReasonGroup"])
    summary = {
        "Split": split,
        "ExcludeQualityIssues": exclude_quality_issues,
        "ExcludeSaturated": exclude_saturated,
        "QualityAuditPath": str(q_path),
        "AnomalyAuditPath": str(a_path),
        "ExcludedUniqueSampleIds": len(excluded),
        "ExcludedSampleIds": sorted(excluded),
    }
    return excluded, excl_df, summary


def select_reference_fixed_sample_ids(
    *,
    specs: list[ModelSpec],
    fixed_count: int,
    reference_model: str,
    reference_config: str,
    split: str,
    step13,
    excluded_sample_ids: set[str] | None = None,
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

    excluded_sample_ids = excluded_sample_ids or set()

    if ref.case_set_path.exists():
        cases = pd.read_csv(ref.case_set_path)
        cases = ensure_sample_id_column(cases)
        if "CaseGroup" in cases.columns:
            fixed = cases[cases["CaseGroup"].astype(str).eq("FixedComparisonCases")].copy()
            fixed = fixed[~fixed["SampleId"].astype(str).isin(excluded_sample_ids)].copy()
            if len(fixed) > 0:
                if "Order" in fixed.columns:
                    fixed = fixed.sort_values("Order")
                return fixed["SampleId"].astype(str).head(fixed_count).tolist()

    metrics = load_metrics_for_spec(ref, split, step13)
    metrics = metrics[~metrics["SampleId"].astype(str).isin(excluded_sample_ids)].copy()
    if "GroundTruthPixels" in metrics.columns:
        metrics = metrics[metrics["GroundTruthPixels"] > 0].copy()
    metrics = metrics.sort_values(["GroundTruthPixels", "SplitOrder"], ascending=[False, True])
    return metrics["SampleId"].astype(str).head(fixed_count).tolist()


def select_individual_cases(
    *,
    specs: list[ModelSpec],
    groups: list[str],
    per_group: int,
    individual_models: list[str] | None = None,
    excluded_sample_ids: set[str] | None = None,
) -> list[tuple[ModelSpec, str, str]]:
    selected: list[tuple[ModelSpec, str, str]] = []
    excluded_sample_ids = excluded_sample_ids or set()

    # En la versión final se generan paneles individuales para los cuatro modelos.
    # Si el usuario pasa --IndividualModels, se usa esa lista; si no, se usan todos
    # los modelos encontrados en specs, manteniendo el orden experimental.
    if individual_models is None or len(individual_models) == 0 or "All" in individual_models:
        preferred_models = {s.model_name for s in specs}
    else:
        preferred_models = {m.strip() for m in individual_models if m.strip()}

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
            sub = sub[~sub["SampleId"].astype(str).isin(excluded_sample_ids)].copy()
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
        "Country",
        "Date",
        "Location",
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
        "Country": "País",
        "Date": "Fecha",
        "Location": "Lugar",
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
    parser.add_argument(
        "--IndividualModels",
        default="All",
        help="Modelos para paneles individuales. Use All o una lista separada por comas: SimpleUNet,EnhancedUNet,TransformerUNet,TransformerPlus.",
    )
    parser.add_argument("--IndividualPerGroup", type=int, default=1)
    parser.add_argument("--NoRawContext", action="store_true")
    parser.add_argument(
        "--QualityAuditPath",
        default=None,
        help="CSV de Step19SampleQualityAudit. Por defecto usa <OutputDir>/Tables/Step19SampleQualityAudit.csv si existe.",
    )
    parser.add_argument(
        "--AnomalyAuditPath",
        default=None,
        help="CSV de Step19MetricsAnomalyAudit. Por defecto usa <OutputDir>/Tables/Step19MetricsAnomalyAudit.csv si existe.",
    )
    parser.add_argument(
        "--NoExcludeQualityIssues",
        action="store_true",
        help="No filtra casos con Target/Reference/productos sospechosos para visualización.",
    )
    parser.add_argument(
        "--NoExcludeSaturated",
        action="store_true",
        help="No filtra casos saturados para visualización.",
    )

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
    individual_models = [x.strip() for x in args.IndividualModels.split(",") if x.strip()]

    specs = build_model_specs(
        project_root=project_root,
        run_tags=run_tags,
        feature_configs=feature_configs,
        threshold_override=args.ThresholdOverride,
    )

    print("\n=== STEP18 START ===")
    print(f"Version: {STEP18_VERSION}")
    print(f"ProjectRoot: {project_root}")
    print(f"OutputDir: {output_dir}")
    print(f"Device: {device}")
    print("Experimentos encontrados:")
    for s in specs:
        print(f"- {s.run_tag} | {s.model_name} | {s.feature_config} | {s.run_id} | threshold={s.threshold:.3f}")

    excluded_sample_ids, exclusion_df, exclusion_summary = build_exclusion_registry(
        output_dir=output_dir,
        split=args.Split,
        project_root=project_root,
        exclude_quality_issues=not args.NoExcludeQualityIssues,
        exclude_saturated=not args.NoExcludeSaturated,
        quality_audit_path=Path(args.QualityAuditPath) if args.QualityAuditPath else None,
        anomaly_audit_path=Path(args.AnomalyAuditPath) if args.AnomalyAuditPath else None,
    )
    exclusion_path = tables_dir / f"Step18ExcludedVisualSampleIds_{args.Split}.csv"
    exclusion_summary_path = logs_dir / f"Step18ExcludedVisualSampleIds_{args.Split}.json"
    if len(exclusion_df) > 0:
        exclusion_df.to_csv(exclusion_path, index=False)
    else:
        pd.DataFrame(columns=["Split", "SampleId", "ReasonGroup", "Reasons", "Country", "Date", "Location"]).to_csv(exclusion_path, index=False)
    exclusion_summary_path.write_text(json.dumps(exclusion_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Visual exclusion filter: {len(excluded_sample_ids)} SampleIds excluded for split {args.Split}")
    print(f"VisualExclusionCSV: {exclusion_path}")

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
        excluded_sample_ids=excluded_sample_ids,
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
                "Description": "Comparación fija 5x2 por modelo, configuración y contexto",
            }
        )
        print(f"[FIG] {out_path}")

    # 2) Paneles individuales 2x3 para casos representativos.
    individual_specs = select_individual_cases(
        specs=specs,
        groups=individual_groups,
        per_group=args.IndividualPerGroup,
        individual_models=individual_models,
        excluded_sample_ids=excluded_sample_ids,
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
        "IndividualModels": individual_models,
        "IndividualPerGroup": args.IndividualPerGroup,
        "RawContextAvailable": dataset is not None,
        "ExcludedVisualSampleIds": sorted(excluded_sample_ids),
        "ExcludedVisualSampleIdsCount": len(excluded_sample_ids),
        "ModelSpecs": [s.__dict__ | {"run_root": str(s.run_root), "feature_dir": str(s.feature_dir), "model_root": str(s.model_root), "metrics_path": str(s.metrics_path), "case_set_path": str(s.case_set_path)} for s in specs],
    }
    (logs_dir / "Step18RunConfig.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== STEP18 COMPLETED ===")
    print(f"FiguresDir: {figures_dir}")
    print(f"MetricsCSV: {metrics_path}")
    print(f"FiguresIndexCSV: {figures_path}")
    print(f"LatexTable: {latex_path}")
    print(f"VisualExclusionCSV: {exclusion_path}")


if __name__ == "__main__":
    main()
