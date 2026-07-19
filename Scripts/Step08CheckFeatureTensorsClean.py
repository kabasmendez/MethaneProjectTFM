#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step08CheckFeatureTensorsClean.py

Valida tensores construidos por Step07BuildFeaturesClean.py para ConfigB o ConfigC.

Verifica:
- existencia de Features.npy y Masks.npy por split
- shapes esperados
- número de canales según Configs/<FeatureConfig>.yaml
- valores finitos
- rangos básicos
- máscaras binarias
- presencia de píxeles positivos
- canales de viento en ConfigC

Uso:

python Scripts/Step08CheckFeatureTensorsClean.py \
  --RunTag Exp271431 \
  --FeatureConfig ConfigC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPLITS = ["Train", "Validation", "Test"]

EXPECTED_SAMPLES_DEFAULT = {
    "Train": 2463,
    "Validation": 528,
    "Test": 528,
}

WIND_FEATURES = ["WindSpeed10m", "WindDirCos10m", "WindDirSin10m"]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe YAML: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"{path} no contiene diccionario YAML válido.")

    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_feature_config(feature_config_name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "Configs" / f"{feature_config_name}.yaml"
    config = load_yaml(path)

    features = config.get("Features")
    if not isinstance(features, list) or len(features) == 0:
        raise ValueError(f"{path}: no contiene lista Features válida.")

    features = [str(x) for x in features]
    input_channels = int(config.get("InputChannels", len(features)))

    if input_channels != len(features):
        raise ValueError(
            f"{path}: InputChannels={input_channels}, pero len(Features)={len(features)}"
        )

    if feature_config_name == "ConfigC":
        missing = [f for f in WIND_FEATURES if f not in features]
        if missing:
            raise ValueError(f"ConfigC no contiene features de viento: {missing}")

    config["Features"] = features
    config["InputChannels"] = input_channels

    return config


def is_binary_mask(y: np.ndarray) -> bool:
    values = np.unique(y)
    return set(values.tolist()).issubset({0, 1})


def summarize_channel(x: np.ndarray, channel_index: int, feature_name: str) -> dict[str, Any]:
    c = x[:, channel_index, :, :]

    return {
        "FeatureName": feature_name,
        "ChannelIndex": int(channel_index),
        "Min": float(np.nanmin(c)),
        "Max": float(np.nanmax(c)),
        "Mean": float(np.nanmean(c)),
        "Std": float(np.nanstd(c)),
        "Finite": bool(np.isfinite(c).all()),
    }


def validate_split(
    *,
    run_tag: str,
    feature_config_name: str,
    feature_names: list[str],
    expected_channels: int,
    split: str,
    expected_samples: int | None,
    strict_samples: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    feature_dir = PROJECT_ROOT / "Outputs" / "Experiments" / run_tag / feature_config_name / "Features"

    x_path = feature_dir / f"{split}Features.npy"
    y_path = feature_dir / f"{split}Masks.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"No existe X: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"No existe Y: {y_path}")

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    if x.ndim != 4:
        raise ValueError(f"{split}: X debe ser N,C,H,W. Recibido {x.shape}")

    if y.ndim != 4:
        raise ValueError(f"{split}: Y debe ser N,1,H,W. Recibido {y.shape}")

    n, c, h, w = x.shape

    if c != expected_channels:
        raise ValueError(
            f"{split}: X tiene {c} canales, pero {feature_config_name} espera {expected_channels}"
        )

    if y.shape != (n, 1, h, w):
        raise ValueError(
            f"{split}: Y shape {y.shape} incompatible con X shape {x.shape}"
        )

    if expected_samples is not None and strict_samples and n != expected_samples:
        raise ValueError(
            f"{split}: muestras={n}, esperado={expected_samples}. "
            "Si estás validando smoke test, usa --NoStrictSamples."
        )

    if not np.isfinite(x).all():
        raise ValueError(f"{split}: X contiene NaN o Inf.")

    if not is_binary_mask(y):
        raise ValueError(f"{split}: Y no es binaria. Valores únicos: {np.unique(y)[:20]}")

    positive_pixels = int(y.sum())

    if positive_pixels <= 0:
        raise ValueError(f"{split}: Y no contiene píxeles positivos.")

    summary = {
        "RunTag": run_tag,
        "FeatureConfig": feature_config_name,
        "Split": split,
        "FeaturePath": str(x_path),
        "MaskPath": str(y_path),
        "Samples": int(n),
        "ExpectedSamples": None if expected_samples is None else int(expected_samples),
        "Channels": int(c),
        "ExpectedChannels": int(expected_channels),
        "Height": int(h),
        "Width": int(w),
        "Xdtype": str(x.dtype),
        "Ydtype": str(y.dtype),
        "XFinite": bool(np.isfinite(x).all()),
        "YBinary": bool(is_binary_mask(y)),
        "MaskPositivePixels": positive_pixels,
        "XMin": float(np.nanmin(x)),
        "XMax": float(np.nanmax(x)),
        "XMean": float(np.nanmean(x)),
        "XStd": float(np.nanstd(x)),
    }

    channel_rows = []
    for idx, feature_name in enumerate(feature_names):
        row = summarize_channel(x, idx, feature_name)
        row.update(
            {
                "RunTag": run_tag,
                "FeatureConfig": feature_config_name,
                "Split": split,
            }
        )
        channel_rows.append(row)

    if feature_config_name == "ConfigC":
        wind_indices = {
            name: feature_names.index(name)
            for name in WIND_FEATURES
        }

        wind_speed = x[:, wind_indices["WindSpeed10m"], :, :]
        wind_cos = x[:, wind_indices["WindDirCos10m"], :, :]
        wind_sin = x[:, wind_indices["WindDirSin10m"], :, :]

        if float(np.nanmin(wind_speed)) < 0:
            raise ValueError(f"{split}: WindSpeed10m contiene valores negativos.")

        if float(np.nanmin(wind_cos)) < -1.0001 or float(np.nanmax(wind_cos)) > 1.0001:
            raise ValueError(f"{split}: WindDirCos10m fuera de [-1,1].")

        if float(np.nanmin(wind_sin)) < -1.0001 or float(np.nanmax(wind_sin)) > 1.0001:
            raise ValueError(f"{split}: WindDirSin10m fuera de [-1,1].")

        summary.update(
            {
                "WindSpeedMin": float(np.nanmin(wind_speed)),
                "WindSpeedMax": float(np.nanmax(wind_speed)),
                "WindSpeedMean": float(np.nanmean(wind_speed)),
                "WindDirCosMin": float(np.nanmin(wind_cos)),
                "WindDirCosMax": float(np.nanmax(wind_cos)),
                "WindDirSinMin": float(np.nanmin(wind_sin)),
                "WindDirSinMax": float(np.nanmax(wind_sin)),
            }
        )

    return summary, channel_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida tensores de features para ConfigB o ConfigC."
    )

    parser.add_argument(
        "--RunTag",
        required=True,
        help="Identificador del experimento. Ejemplo: Exp271431.",
    )

    parser.add_argument(
        "--FeatureConfig",
        default="ConfigB",
        choices=["ConfigB", "ConfigC"],
        help="Configuración de features a validar.",
    )

    parser.add_argument(
        "--NoStrictSamples",
        action="store_true",
        help="No exige número exacto de muestras por split. Útil para smoke tests.",
    )

    args = parser.parse_args()

    run_tag = args.RunTag
    feature_config_name = args.FeatureConfig
    strict_samples = not args.NoStrictSamples

    feature_config = load_feature_config(feature_config_name)
    feature_names = feature_config["Features"]
    expected_channels = int(feature_config["InputChannels"])

    output_root = PROJECT_ROOT / "Outputs" / "Experiments" / run_tag / feature_config_name
    tables_dir = output_root / "Tables"
    audit_dir = output_root / "Audit"

    tables_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("=== STEP08 CHECK FEATURE TENSORS START ===")
    print(f"RunTag: {run_tag}")
    print(f"FeatureConfig: {feature_config_name}")
    print(f"ExpectedChannels: {expected_channels}")
    print(f"Features: {feature_names}")
    print(f"StrictSamples: {strict_samples}")

    summaries = []
    channel_rows = []

    for split in SPLITS:
        expected_samples = EXPECTED_SAMPLES_DEFAULT.get(split)

        summary, rows = validate_split(
            run_tag=run_tag,
            feature_config_name=feature_config_name,
            feature_names=feature_names,
            expected_channels=expected_channels,
            split=split,
            expected_samples=expected_samples,
            strict_samples=strict_samples,
        )

        summaries.append(summary)
        channel_rows.extend(rows)

        print("")
        print(f"{split}: OK")
        print(f"  X: ({summary['Samples']}, {summary['Channels']}, {summary['Height']}, {summary['Width']})")
        print(f"  Y positive pixels: {summary['MaskPositivePixels']}")
        print(f"  X min/max: {summary['XMin']:.4f} / {summary['XMax']:.4f}")

        if feature_config_name == "ConfigC":
            print(
                "  WindSpeed min/max/mean: "
                f"{summary['WindSpeedMin']:.4f} / "
                f"{summary['WindSpeedMax']:.4f} / "
                f"{summary['WindSpeedMean']:.4f}"
            )
            print(
                "  WindCos min/max: "
                f"{summary['WindDirCosMin']:.4f} / "
                f"{summary['WindDirCosMax']:.4f}"
            )
            print(
                "  WindSin min/max: "
                f"{summary['WindDirSinMin']:.4f} / "
                f"{summary['WindDirSinMax']:.4f}"
            )

    summary_df = pd.DataFrame(summaries)
    channel_df = pd.DataFrame(channel_rows)

    summary_path = tables_dir / "FeatureTensorCheckSummary.csv"
    channel_path = tables_dir / "FeatureChannelStats.csv"
    audit_path = audit_dir / "FeatureTensorCheckAudit.json"

    summary_df.to_csv(summary_path, index=False)
    channel_df.to_csv(channel_path, index=False)

    save_json(
        audit_path,
        {
            "RunTag": run_tag,
            "FeatureConfig": feature_config_name,
            "ExpectedChannels": expected_channels,
            "FeatureNames": feature_names,
            "StrictSamples": strict_samples,
            "SummaryPath": str(summary_path),
            "ChannelStatsPath": str(channel_path),
            "AllSplitsPassed": True,
        },
    )

    print("")
    print("=== STEP08 CHECK FEATURE TENSORS COMPLETED ===")
    print(f"Summary: {summary_path}")
    print(f"ChannelStats: {channel_path}")
    print(f"Audit: {audit_path}")
    print("")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
