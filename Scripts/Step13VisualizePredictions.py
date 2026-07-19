#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step13VisualizePredictions.py

Genera figuras cualitativas completas para análisis y documento.

Compatible con ConfigB y ConfigC.

Layout 3x4:
Fila 1:
  1. Target SWIR2-NIR-Azul
  2. Reference SWIR2-NIR-Azul
  3. Target B11
  4. Target B12

Fila 2:
  5. Ground truth
  6. MBMP+
  7. CH4 si está disponible
  8. Predicted probability + GT + predicción + viento

Fila 3:
  9. Predicted mask + GT + viento
  10. Overlay TP/FP/FN
  11. Leyenda
  12. Métricas

Convenciones:
- Ground truth: rojo grueso
- Predicción: cian grueso
- Viento: naranja fuerte desde centroide de GT
- TP: verde
- FP: naranja/amarillo
- FN: rojo
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Source.Models.ModelFactory import CreateModel

try:
    from Source.TacoIndex import LoadTacoDataset, GetSampleTable
    from Source.ReadTacoSample import ReadFullTacoSample
    HAS_TACO_READERS = True
except Exception:
    HAS_TACO_READERS = False


FEATURE_READY_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}

COLORS = {
    "navy": "#4883F1",
    "blue": "#36B0F1",
    "cyan": "#00D4FF",
    "green": "#1F9D55",
    "red": "#F40E0A",
    "orange": "#3DFF87",
    "wind": "#269B7E",
    "yellow": "#EBF76D",
    "gray": "#E9EEF5",
    "darkgray": "#536271",
    "text": "#172B4D",
    "white": "#FFFFFF",
    "black": "#000000",
}

CH4_ROSE_CMAP = LinearSegmentedColormap.from_list(
    "ch4_sequential_rose",
    ["#fff5f7", "#efd3d8", "#f3c2cd", "#ef7694", "#f11755"],
    N=256,
)

BAND_INDEX = {
    "B1": 0,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B8A": 8,
    "B9": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Montserrat",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["blue"],
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "savefig.facecolor": "white",
        }
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe YAML: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"{path} no contiene dict YAML.")

    return data


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_nested(config: dict[str, Any], paths: list[list[str]]) -> Any:
    for path in paths:
        current: Any = config
        ok = True

        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]

        if ok and current is not None:
            return current

    return None


def resolve_dataset_location(project_config: dict[str, Any]) -> tuple[str | None, str | None]:
    dataset_path = get_nested(
        project_config,
        [
            ["DatasetPath"],
            ["Dataset", "DatasetPath"],
            ["TacoDatasetPath"],
            ["Dataset", "TacoDatasetPath"],
        ],
    )

    if dataset_path is not None:
        path = Path(str(dataset_path))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path), None

    data_root = get_nested(
        project_config,
        [
            ["DataRoot"],
            ["DatasetRoot"],
            ["TacoRoot"],
            ["TacoDatasetRoot"],
            ["Dataset", "DataRoot"],
            ["Dataset", "DatasetRoot"],
            ["Dataset", "TacoRoot"],
            ["Dataset", "TacoDatasetRoot"],
        ],
    )

    dataset_name = get_nested(
        project_config,
        [
            ["DatasetName"],
            ["TacoDatasetName"],
            ["Dataset", "DatasetName"],
            ["Dataset", "TacoDatasetName"],
            ["Dataset", "Name"],
        ],
    )

    if data_root is None:
        return None, None

    return str(data_root), None if dataset_name is None else str(dataset_name)


def ensure_sample_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SampleId" in df.columns:
        df["SampleId"] = df["SampleId"].astype(str)
        return df

    for candidate in ["sample_id", "id", "Id", "ID", "sampleId", "SampleID"]:
        if candidate in df.columns:
            df["SampleId"] = df[candidate].astype(str)
            return df

    raise KeyError(f"No encontré SampleId/id. Columnas: {list(df.columns)}")


def load_dataset_flexible(data_root: str, dataset_name: str | None):
    attempts = []

    if dataset_name is not None:
        attempts.extend(
            [
                lambda: LoadTacoDataset(data_root, dataset_name),
                lambda: LoadTacoDataset(DataRoot=data_root, DatasetName=dataset_name),
                lambda: LoadTacoDataset(Path(data_root) / dataset_name),
                lambda: LoadTacoDataset(str(Path(data_root) / dataset_name)),
            ]
        )

    attempts.extend(
        [
            lambda: LoadTacoDataset(data_root),
            lambda: LoadTacoDataset(Path(data_root)),
            lambda: LoadTacoDataset(DataRoot=data_root),
        ]
    )

    last_error = None

    for attempt in attempts:
        try:
            result = attempt()
            dataset = result[0] if isinstance(result, tuple) else result
            sample_table = GetSampleTable(dataset)

            if isinstance(sample_table, pd.DataFrame):
                return dataset, ensure_sample_id_column(sample_table)

        except Exception as exc:
            last_error = exc

    print(f"WARNING: no pude cargar dataset raw. Último error: {repr(last_error)}")
    return None, None


def read_sample_flexible(dataset: Any, sample_table: pd.DataFrame, sample_id: str) -> dict[str, Any] | None:
    if dataset is None or sample_table is None:
        return None

    attempts = [
        lambda: ReadFullTacoSample(Dataset=dataset, SampleTable=sample_table, SampleId=sample_id),
        lambda: ReadFullTacoSample(dataset, sample_table, sample_id),
        lambda: ReadFullTacoSample(Dataset=dataset, SampleId=sample_id),
        lambda: ReadFullTacoSample(dataset, sample_id),
    ]

    last_error = None

    for attempt in attempts:
        try:
            sample = attempt()
            if isinstance(sample, dict):
                return sample
        except Exception as exc:
            last_error = exc

    print(f"WARNING: no pude leer muestra raw {sample_id}. Último error: {repr(last_error)}")
    return None


def load_feature_names(feature_config: str) -> list[str]:
    path = PROJECT_ROOT / "Configs" / f"{feature_config}.yaml"
    cfg = load_yaml(path)
    features = cfg.get("Features")

    if not isinstance(features, list):
        raise ValueError(f"{path}: falta lista Features.")

    return [str(x) for x in features]


def load_split_index(run_root: Path, split: str, expected_n: int) -> pd.DataFrame:
    path = run_root / "Tables" / FEATURE_READY_FILES[split]

    if not path.exists():
        return pd.DataFrame(
            {
                "SampleId": [str(i) for i in range(expected_n)],
                "SplitOrder": np.arange(expected_n, dtype=int),
            }
        )

    df = pd.read_csv(path)
    df = ensure_sample_id_column(df)
    df["SplitOrder"] = np.arange(len(df), dtype=int)

    if len(df) != expected_n:
        if len(df) > expected_n:
            df = df.head(expected_n).copy()
        else:
            extra = pd.DataFrame(
                {
                    "SampleId": [str(i) for i in range(len(df), expected_n)],
                    "SplitOrder": np.arange(len(df), expected_n, dtype=int),
                }
            )
            df = pd.concat([df[["SampleId", "SplitOrder"]], extra], ignore_index=True)

    return df[["SampleId", "SplitOrder"]].copy()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def extract_checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {path}")

    checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint no es dict. Tipo: {type(checkpoint)}")

    return checkpoint


def get_state_dict_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ["model_state_dict", "state_dict", "model", "model_state"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]

    raise KeyError(f"No encontré state dict. Llaves: {list(checkpoint.keys())}")


def get_model_parameters(checkpoint: dict[str, Any], training_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(checkpoint.get("model_parameters"), dict):
        return dict(checkpoint["model_parameters"])

    if isinstance(checkpoint.get("ModelParameters"), dict):
        return dict(checkpoint["ModelParameters"])

    if isinstance(training_config.get("ModelParameters"), dict):
        return dict(training_config["ModelParameters"])

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


def to_chw(arr: Any) -> np.ndarray | None:
    if arr is None:
        return None

    a = np.asarray(arr)

    if a.ndim == 2:
        return a[None, :, :]

    if a.ndim != 3:
        return None

    if a.shape[0] <= 30:
        return a

    if a.shape[-1] <= 30:
        return np.moveaxis(a, -1, 0)

    return None


def get_sample_array(sample: dict[str, Any] | None, key: str) -> np.ndarray | None:
    if sample is None:
        return None

    aliases = {
        "Target": ["Target", "target"],
        "Reference": ["Reference", "reference"],
        "CH4": [
            "CH4",
            "ch4",
            "Ch4",
            "Methane",
            "methane",
            "CH4Enhancement",
            "ch4_enhancement",
            "PlumeConcentration",
            "plume_concentration",
            "Concentration",
            "concentration",
        ],
    }

    for candidate in aliases.get(key, [key]):
        if candidate in sample:
            return np.asarray(sample[candidate])

    if key == "CH4":
        for candidate in sample.keys():
            ck = str(candidate).lower()
            if "ch4" in ck or "methane" in ck:
                try:
                    return np.asarray(sample[candidate])
                except Exception:
                    pass

    return None


def get_band_from_cube(cube: Any, band_name: str) -> np.ndarray | None:
    chw = to_chw(cube)

    if chw is None:
        return None

    idx = BAND_INDEX.get(band_name)

    if idx is None or idx >= chw.shape[0]:
        return None

    return np.asarray(chw[idx], dtype=np.float32)


def get_feature_channel(x: np.ndarray, feature_names: list[str], name: str) -> np.ndarray | None:
    if name not in feature_names:
        return None
    return np.asarray(x[feature_names.index(name)], dtype=np.float32)


def require_mbmp_base(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, str]:
    for name in ["MBMPPlus", "MBMP"]:
        channel = get_feature_channel(x, feature_names, name)
        if channel is not None:
            return channel, name

    raise KeyError(
        "No encontré MBMPPlus ni MBMP en feature_names. "
        f"Features disponibles: {feature_names}"
    )


def normalize_for_display(arr: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(finite, [pmin, pmax])

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def get_band_or_feature(
    cube: Any,
    x: np.ndarray,
    feature_names: list[str],
    band_name: str,
) -> np.ndarray | None:
    band = get_band_from_cube(cube, band_name)

    if band is not None:
        return band

    return get_feature_channel(x, feature_names, band_name)


def make_swir_nir_blue_composite(
    cube: Any,
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, str]:
    r = get_band_or_feature(cube, x, feature_names, "B12")
    g = get_band_or_feature(cube, x, feature_names, "B8A")
    b = get_band_or_feature(cube, x, feature_names, "B2")

    title = "SWIR2-NIR-Azul\nB12-B8A-B2"

    if b is None:
        b = get_band_or_feature(cube, x, feature_names, "B11")
        title = "SWIR2-NIR-SWIR1\nB12-B8A-B11"

    if r is not None and g is not None and b is not None:
        rgb = np.stack(
            [
                normalize_for_display(r),
                normalize_for_display(g),
                normalize_for_display(b),
            ],
            axis=-1,
        )
        return rgb, title

    fallback_order = ["B12", "B11", "B8A", "MBMPPlus", "MBMP"]
    channels = []

    for name in fallback_order:
        c = get_feature_channel(x, feature_names, name)
        if c is not None:
            channels.append(normalize_for_display(c))
        if len(channels) == 3:
            break

    if len(channels) == 3:
        return np.stack(channels, axis=-1), "Fallback features"

    gray = normalize_for_display(x[0])
    return np.stack([gray, gray, gray], axis=-1), "Fallback canal 0"


def get_wind_values(x: np.ndarray, feature_names: list[str]) -> tuple[float | None, float | None, float | None]:
    speed = get_feature_channel(x, feature_names, "WindSpeed10m")
    cos = get_feature_channel(x, feature_names, "WindDirCos10m")
    sin = get_feature_channel(x, feature_names, "WindDirSin10m")

    if speed is None or cos is None or sin is None:
        return None, None, None

    return float(np.nanmean(speed)), float(np.nanmean(cos)), float(np.nanmean(sin))


def get_plume_centroid(mask: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(mask > 0)

    if coords.shape[0] == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0

    rows = coords[:, 0]
    cols = coords[:, 1]

    return float(cols.mean()), float(rows.mean())


def safe_contour(
    ax: plt.Axes,
    mask: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "solid",
) -> bool:
    mask = np.asarray(mask)

    if mask.ndim != 2:
        return False

    if np.nanmax(mask) <= 0:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contour(
            mask,
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
            linestyles=linestyle,
        )

    return True


def draw_wind_arrow(
    ax: plt.Axes,
    *,
    mask: np.ndarray,
    wind_cos: float | None,
    wind_sin: float | None,
    color: str = COLORS["wind"],
) -> None:
    if wind_cos is None or wind_sin is None:
        return

    if not np.isfinite(wind_cos) or not np.isfinite(wind_sin):
        return

    h, w = mask.shape
    cx, cy = get_plume_centroid(mask)

    length = 0.23 * min(h, w)
    dx = length * float(wind_cos)
    dy = -length * float(wind_sin)

    ax.scatter(
        [cx],
        [cy],
        s=28,
        color=COLORS["white"],
        edgecolor=COLORS["navy"],
        linewidth=1.0,
        zorder=12,
    )

    ax.arrow(
        cx,
        cy,
        dx,
        dy,
        color=color,
        width=1.5,
        head_width=7,
        length_includes_head=True,
        zorder=13,
    )


@torch.no_grad()
def predict_one(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.array(x[None, ...], dtype=np.float32, copy=True)).to(device)

    logits = model(tensor)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = (prob >= float(threshold)).astype(np.uint8)

    return prob.astype(np.float32), pred


def compute_masks(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target)
    fp = np.logical_and(pred, np.logical_not(target))
    fn = np.logical_and(np.logical_not(pred), target)

    return tp, fp, fn


def make_overlay_rgb(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h, w = target.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    tp, fp, fn = compute_masks(pred, target)

    rgb[tp] = np.array([0.12, 0.62, 0.34])  # green
    rgb[fp] = np.array([1.00, 0.58, 0.00])  # orange/yellow
    rgb[fn] = np.array([0.90, 0.08, 0.08])  # red

    return rgb


def add_no_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def panel_empty(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["darkgray"],
        transform=ax.transAxes,
    )
    add_no_ticks(ax)


def plot_case(
    *,
    x: np.ndarray,
    y: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    feature_names: list[str],
    sample: dict[str, Any] | None,
    row: pd.Series,
    output_path: Path,
    threshold: float,
) -> None:

    target_cube = get_sample_array(sample, "Target")
    reference_cube = get_sample_array(sample, "Reference")
    ch4_arr = get_sample_array(sample, "CH4")

    target_composite, target_title = make_swir_nir_blue_composite(
        target_cube,
        x,
        feature_names,
    )

    reference_composite, reference_title = make_swir_nir_blue_composite(
        reference_cube,
        x,
        feature_names,
    )

    target_b11 = get_band_or_feature(target_cube, x, feature_names, "B11")
    target_b12 = get_band_or_feature(target_cube, x, feature_names, "B12")

    mbmp_plus, mbmp_name = require_mbmp_base(x, feature_names)

    gt = (y[0] > 0).astype(np.uint8)
    pred_mask = (pred > 0).astype(np.uint8)

    wind_speed, wind_cos, wind_sin = get_wind_values(x, feature_names)

    fig, axes = plt.subplots(3, 4, figsize=(18.0, 12.2))
    axes = axes.ravel()

    title = (
        f"{row['CaseGroup']} | SampleId={row['SampleId']} | "
        f"Dice={row.get('Dice', np.nan):.3f} | IoU={row.get('IoU', np.nan):.3f}"
    )

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.985,
    )

    # Fila 1: contexto
    ax = axes[0]
    ax.imshow(target_composite)
    ax.set_title(f"Target\n{target_title}")
    add_no_ticks(ax)

    ax = axes[1]
    ax.imshow(reference_composite)
    ax.set_title(f"Reference\n{reference_title}")
    add_no_ticks(ax)

    ax = axes[2]
    if target_b11 is not None:
        ax.imshow(normalize_for_display(target_b11), cmap="gray", vmin=0, vmax=1)
        safe_contour(ax, gt, color=COLORS["red"], linewidth=2.7)
        ax.set_title("Target B11\nGT rojo")
        add_no_ticks(ax)
    else:
        panel_empty(ax, "Target B11", "No disponible")

    ax = axes[3]
    if target_b12 is not None:
        ax.imshow(normalize_for_display(target_b12), cmap="gray", vmin=0, vmax=1)
        safe_contour(ax, gt, color=COLORS["red"], linewidth=2.7)
        ax.set_title("Target B12\nGT rojo")
        add_no_ticks(ax)
    else:
        panel_empty(ax, "Target B12", "No disponible")

    # Fila 2: GT, MBMP+, CH4, probabilidad
    ax = axes[4]
    ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
    safe_contour(ax, gt, color=COLORS["red"], linewidth=3.2)
    ax.set_title(f"Ground truth\npixels={int(gt.sum())}")
    add_no_ticks(ax)

    ax = axes[5]
    im = ax.imshow(normalize_for_display(mbmp_plus), cmap="seismic", vmin=0, vmax=2)
    safe_contour(ax, gt, color=COLORS["red"], linewidth=2.7)
    ax.set_title(f"{mbmp_name}\nGT rojo")
    add_no_ticks(ax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[6]
    if ch4_arr is not None:
        ch4_chw = to_chw(ch4_arr)
        if ch4_chw is not None:
            ch4_map = ch4_chw[0]
        else:
            ch4_map = np.asarray(ch4_arr)
            if ch4_map.ndim == 3:
                ch4_map = ch4_map[..., 0]

        im = ax.imshow(normalize_for_display(ch4_map), cmap='GnBu', vmin=0, vmax=1)
        safe_contour(ax, gt, color=COLORS["red"], linewidth=2.7)
        ax.set_title("CH4\nGT rojo")
        add_no_ticks(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        panel_empty(ax, "CH4", "No disponible\nen raw sample")

    ax = axes[7]
    im = ax.imshow(prob, cmap="viridis", vmin=0, vmax=1)
    # safe_contour(ax, gt, color=COLORS["red"], linewidth=2.8)
    safe_contour(ax, pred_mask, color=COLORS["cyan"], linewidth=2.8)
    draw_wind_arrow(ax, mask=gt, wind_cos=wind_cos, wind_sin=wind_sin)
    ax.set_title("Predicted probability\nGT rojo, pred cian")
    add_no_ticks(ax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Fila 3: pred, overlay, leyenda, métricas
    ax = axes[8]
    pred_canvas = np.zeros_like(pred_mask, dtype=np.float32)
    ax.imshow(pred_canvas, cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap="winter", alpha=0.60)
    # safe_contour(ax, gt, color=COLORS["red"], linewidth=2.8)
    safe_contour(ax, pred_mask, color=COLORS["cyan"], linewidth=3.0)
    draw_wind_arrow(ax, mask=gt, wind_cos=wind_cos, wind_sin=wind_sin)
    ax.set_title("Predicted mask")
    add_no_ticks(ax)

    ax = axes[9]
    ax.imshow(make_overlay_rgb(gt, pred_mask), vmin=0, vmax=1)
    # safe_contour(ax, gt, color=COLORS["red"], linewidth=2.3)
    safe_contour(ax, pred_mask, color=COLORS["cyan"], linewidth=2.3)
    ax.set_title("Overlay TP / FP / FN")
    add_no_ticks(ax)

    ax = axes[10]
    ax.axis("off")
    legend_handles = [
        mpatches.Patch(color=COLORS["red"], label="Ground truth: pluma real"),
        mpatches.Patch(color=COLORS["cyan"], label="Predicción: pluma estimada"),
        mpatches.Patch(color=COLORS["wind"], label="Vector de viento"),
        mpatches.Patch(color=COLORS["green"], label="TP: pluma correctamente detectada"),
        mpatches.Patch(color=COLORS["yellow"], label="FP: falsa alarma"),
        mpatches.Patch(color=COLORS["red"], label="FN: pluma omitida"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="center",
        fontsize=12,
        frameon=True,
        title="Convenciones",
        title_fontsize=14,
    )

    ax = axes[11]
    ax.axis("off")

    metrics_lines = [
        "Métricas del caso",
        "",
        f"Dice: {row.get('Dice', np.nan):.4f}",
        f"IoU: {row.get('IoU', np.nan):.4f}",
        f"Precision: {row.get('Precision', np.nan):.4f}",
        f"Recall: {row.get('Recall', np.nan):.4f}",
        "",
        f"GT pixels: {int(row.get('GroundTruthPixels', gt.sum()))}",
        f"Pred pixels: {int(row.get('PredictedPixels', pred_mask.sum()))}",
        f"FP pixels: {int(row.get('FalsePositivePixels', 0))}",
        f"FN pixels: {int(row.get('FalseNegativePixels', 0))}",
    ]

    if wind_speed is not None:
        metrics_lines.extend(
            [
                "",
                "Viento 10 m",
                f"Speed: {wind_speed:.3f}",
                f"cos: {wind_cos:.3f}",
                f"sin: {wind_sin:.3f}",
            ]
        )

    ax.text(
        0.03,
        0.98,
        "\n".join(metrics_lines),
        ha="left",
        va="top",
        fontsize=12,
        color=COLORS["text"],
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor=COLORS["blue"],
            alpha=0.96,
        ),
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.955])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sanitize_filename(value: str) -> str:
    value = str(value)
    for ch in ["/", "\\", ":", " ", "|", ",", ";", "{", "}", "[", "]"]:
        value = value.replace(ch, "_")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualiza predicciones para casos seleccionados.")

    parser.add_argument("--RunTag", required=True)
    parser.add_argument("--FeatureConfig", default="ConfigB", choices=["ConfigB", "ConfigC"])
    parser.add_argument("--ModelName", required=True)
    parser.add_argument("--RunName", required=True)
    parser.add_argument("--Split", default="Test", choices=["Train", "Validation", "Test"])
    parser.add_argument("--Checkpoint", default="BestModel.pt")
    parser.add_argument("--Threshold", type=float, default=0.5)
    parser.add_argument("--MaxCases", type=int, default=None)
    parser.add_argument("--Device", default="auto")
    parser.add_argument("--CaseSetPath", default=None)
    parser.add_argument("--ProjectConfig", default="Configs/ProjectConfig.yaml")
    parser.add_argument("--NoRawContext", action="store_true")

    args = parser.parse_args()

    apply_style()

    device = get_device(args.Device)
    run_id = f"{args.ModelName}_{args.RunName}"

    run_root = PROJECT_ROOT / "Outputs" / "Experiments" / args.RunTag
    feature_dir = run_root / args.FeatureConfig / "Features"
    model_root = run_root / args.FeatureConfig / run_id

    figures_dir = model_root / "Figures" / "PredictionCases"
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(args.FeatureConfig)

    x_path = feature_dir / f"{args.Split}Features.npy"
    y_path = feature_dir / f"{args.Split}Masks.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"No existe X: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"No existe Y: {y_path}")

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    input_channels = int(x.shape[1])
    output_channels = 1

    split_index = load_split_index(run_root, args.Split, expected_n=int(x.shape[0]))
    sample_to_index = {
        str(row["SampleId"]): int(row["SplitOrder"])
        for _, row in split_index.iterrows()
    }

    if args.CaseSetPath is not None:
        case_path = Path(args.CaseSetPath)
    else:
        case_path = model_root / "Tables" / f"VisualizationCaseSet_{run_id}.csv"

    if not case_path.exists():
        raise FileNotFoundError(f"No existe CaseSet: {case_path}. Ejecuta Step12 primero.")

    cases = pd.read_csv(case_path)
    cases["SampleId"] = cases["SampleId"].astype(str)

    if args.MaxCases is not None:
        cases = cases.head(int(args.MaxCases)).copy()

    checkpoint_path = model_root / "Checkpoints" / args.Checkpoint
    checkpoint = extract_checkpoint_payload(checkpoint_path, device=device)
    training_config = load_json_if_exists(model_root / "TrainingConfig.json")
    model_parameters = get_model_parameters(checkpoint, training_config)

    checkpoint_input_channels = checkpoint.get("input_channels", checkpoint.get("InputChannels", None))
    if checkpoint_input_channels is not None and int(checkpoint_input_channels) != input_channels:
        raise ValueError(
            f"Checkpoint input_channels={checkpoint_input_channels}, pero X tiene {input_channels}."
        )

    model = build_model(
        model_name=args.ModelName,
        input_channels=input_channels,
        output_channels=output_channels,
        model_parameters=model_parameters,
    )

    state_dict = get_state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    dataset = None
    sample_table = None

    if not args.NoRawContext and HAS_TACO_READERS:
        project_config_path = Path(args.ProjectConfig)
        if not project_config_path.is_absolute():
            project_config_path = PROJECT_ROOT / project_config_path

        if project_config_path.exists():
            project_config = load_yaml(project_config_path)
            data_root, dataset_name = resolve_dataset_location(project_config)

            if data_root is not None:
                dataset, sample_table = load_dataset_flexible(data_root, dataset_name)
        else:
            print(f"WARNING: no existe ProjectConfig: {project_config_path}")

    print("")
    print("=== VISUALIZE PREDICTIONS START ===")
    print(f"RunTag: {args.RunTag}")
    print(f"FeatureConfig: {args.FeatureConfig}")
    print(f"RunId: {run_id}")
    print(f"Split: {args.Split}")
    print(f"Cases: {len(cases)}")
    print(f"InputChannels: {input_channels}")
    print(f"RawContextAvailable: {dataset is not None}")
    print(f"FiguresDir: {figures_dir}")

    index_rows = []

    for _, row in cases.iterrows():
        sample_id = str(row["SampleId"])

        if sample_id not in sample_to_index:
            raise KeyError(f"SampleId {sample_id} no está en split {args.Split}.")

        idx = sample_to_index[sample_id]

        x_i = np.array(x[idx], dtype=np.float32, copy=True)
        y_i = np.array(y[idx], dtype=np.uint8, copy=True)

        sample = read_sample_flexible(dataset, sample_table, sample_id) if dataset is not None else None

        prob, pred = predict_one(
            model=model,
            x=x_i,
            device=device,
            threshold=float(args.Threshold),
        )

        case_group = sanitize_filename(row["CaseGroup"])
        order = int(row["Order"]) if "Order" in row else len(index_rows) + 1

        file_name = f"{case_group}_{order:02d}_{sanitize_filename(sample_id)}.png"
        out_path = figures_dir / file_name

        plot_case(
            x=x_i,
            y=y_i,
            prob=prob,
            pred=pred,
            feature_names=feature_names,
            sample=sample,
            row=row,
            output_path=out_path,
            threshold=float(args.Threshold),
        )

        index_rows.append(
            {
                "RunTag": args.RunTag,
                "FeatureConfig": args.FeatureConfig,
                "RunId": run_id,
                "Split": args.Split,
                "CaseGroup": row["CaseGroup"],
                "Order": order,
                "SampleId": sample_id,
                "FigurePath": str(out_path),
                "Dice": row.get("Dice", np.nan),
                "IoU": row.get("IoU", np.nan),
                "Precision": row.get("Precision", np.nan),
                "Recall": row.get("Recall", np.nan),
            }
        )

        print("Saved:", out_path)

    index_df = pd.DataFrame(index_rows)
    index_path = figures_dir / "PredictionCaseIndex.csv"
    index_df.to_csv(index_path, index=False)

    save_json(
        figures_dir / "VisualizationRunConfig.json",
        {
            "RunTag": args.RunTag,
            "FeatureConfig": args.FeatureConfig,
            "RunId": run_id,
            "Split": args.Split,
            "Checkpoint": args.Checkpoint,
            "Threshold": float(args.Threshold),
            "InputChannels": input_channels,
            "FeatureNames": feature_names,
            "RawContextAvailable": dataset is not None,
            "Panels": [
                "Target SWIR2-NIR-Blue",
                "Reference SWIR2-NIR-Blue",
                "Target B11",
                "Target B12",
                "Ground truth",
                "MBMPPlus or MBMP",
                "CH4 if available",
                "Predicted probability",
                "Predicted mask",
                "TP/FP/FN overlay",
                "Legend",
                "Metrics",
            ],
        },
    )

    print("")
    print("=== VISUALIZE PREDICTIONS COMPLETED ===")
    print(f"Index: {index_path}")
    print(index_df.to_string(index=False))


if __name__ == "__main__":
    main()
