#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step07BuildFeaturesClean.py

Construye tensores limpios de features para segmentación de plumas de metano.

Soporta:
- ConfigB: 9 canales
- ConfigC: ConfigB + viento en coordenadas polares/cartesianas normalizadas:
    WindSpeed10m
    WindDirCos10m
    WindDirSin10m

Entradas:
- Outputs/Experiments/<RunTag>/Tables/SplitTrainFeatureReady.csv
- Outputs/Experiments/<RunTag>/Tables/SplitValidationFeatureReady.csv
- Outputs/Experiments/<RunTag>/Tables/SplitTestFeatureReady.csv
- Configs/<FeatureConfig>.yaml
- Configs/ProjectConfig.yaml

Salidas:
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/<Split>Features.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Features/<Split>Masks.npy
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Tables/FeatureBuildSummary.csv
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Tables/FeatureBuildSampleSummary.csv
- Outputs/Experiments/<RunTag>/<FeatureConfig>/Audit/FeatureBuildAudit.json

Principio metodológico:
- Las features X se construyen sin usar ground-truth.
- La máscara plume se usa solamente como Y.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Project root e imports locales
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Source.FeatureEngineering import BuildFeatureStack
from Source.ReadTacoSample import ReadFullTacoSample
from Source.TacoIndex import LoadTacoDataset, GetSampleTable


# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------

SPLITS = ["Train", "Validation", "Test"]

FEATURE_READY_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}

WIND_FEATURES = {"WindSpeed10m", "WindDirCos10m", "WindDirSin10m"}


# ---------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el YAML: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"{path} no contiene un diccionario YAML válido.")

    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def as_str_id(value: Any) -> str:
    return str(value).strip()


def ensure_sample_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SampleId" in df.columns:
        df["SampleId"] = df["SampleId"].map(as_str_id)
        return df

    candidates = [
        "sample_id",
        "id",
        "Id",
        "ID",
        "sampleId",
        "SampleID",
    ]

    for col in candidates:
        if col in df.columns:
            df["SampleId"] = df[col].map(as_str_id)
            return df

    raise KeyError(
        "No encontré columna de identificador de muestra. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def find_project_config_path(project_config_arg: str) -> Path:
    path = Path(project_config_arg)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"No existe ProjectConfig: {path}")

    return path


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


def resolve_dataset_location(project_config: dict[str, Any]) -> tuple[str, str | None]:
    """
    Intenta resolver el dataset desde ProjectConfig.yaml con varias estructuras posibles.

    Casos soportados:
    - Dataset:
        DataRoot: ...
        DatasetName: ...
    - DataRoot: ...
      DatasetName: ...
    - TacoDatasetRoot / TacoDatasetName
    - DatasetPath absoluto o relativo.
    """

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
        dataset_path = str(dataset_path)
        path = Path(dataset_path)
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
        raise KeyError(
            "No pude resolver DataRoot/DatasetPath desde ProjectConfig.yaml. "
            "Revisa Configs/ProjectConfig.yaml."
        )

    data_root = str(data_root)
    dataset_name = None if dataset_name is None else str(dataset_name)

    return data_root, dataset_name


def load_dataset_flexible(data_root: str, dataset_name: str | None) -> tuple[Any, pd.DataFrame]:
    """
    Carga TACO usando varias firmas posibles de LoadTacoDataset.
    """

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

    last_error: Exception | None = None

    for attempt in attempts:
        try:
            result = attempt()

            if isinstance(result, tuple):
                dataset = result[0]
            else:
                dataset = result

            sample_table = GetSampleTable(dataset)

            if not isinstance(sample_table, pd.DataFrame):
                raise TypeError(
                    f"GetSampleTable devolvió {type(sample_table)}; se esperaba DataFrame."
                )

            sample_table = ensure_sample_id_column(sample_table)

            return dataset, sample_table

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "No pude cargar el dataset TACO con ninguna firma conocida de LoadTacoDataset. "
        f"Último error: {repr(last_error)}"
    )


def read_sample_flexible(dataset: Any, sample_table: pd.DataFrame, sample_id: str) -> dict[str, Any]:
    """
    Lee una muestra con ReadFullTacoSample usando firmas alternativas.
    """

    attempts = [
        lambda: ReadFullTacoSample(
            Dataset=dataset,
            SampleTable=sample_table,
            SampleId=sample_id,
        ),
        lambda: ReadFullTacoSample(dataset, sample_table, sample_id),
        lambda: ReadFullTacoSample(
            Dataset=dataset,
            SampleId=sample_id,
        ),
        lambda: ReadFullTacoSample(dataset, sample_id),
    ]

    last_error: Exception | None = None

    for attempt in attempts:
        try:
            sample = attempt()

            if not isinstance(sample, dict):
                raise TypeError(
                    f"ReadFullTacoSample devolvió {type(sample)}; se esperaba dict."
                )

            return sample

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"No pude leer SampleId={sample_id} con ReadFullTacoSample. "
        f"Último error: {repr(last_error)}"
    )


# ---------------------------------------------------------------------
# Configuración de features
# ---------------------------------------------------------------------

def load_feature_config(feature_config_name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "Configs" / f"{feature_config_name}.yaml"

    config = load_yaml(path)

    declared_name = config.get("FeatureConfig")
    if declared_name is not None and str(declared_name) != feature_config_name:
        raise ValueError(
            f"{path}: FeatureConfig={declared_name}, pero se pidió {feature_config_name}."
        )

    features = config.get("Features")

    if not isinstance(features, list) or len(features) == 0:
        raise ValueError(f"{path}: no contiene una lista Features válida.")

    features = [str(x) for x in features]
    input_channels = config.get("InputChannels", len(features))

    if int(input_channels) != len(features):
        raise ValueError(
            f"{path}: InputChannels={input_channels}, pero len(Features)={len(features)}."
        )

    if feature_config_name == "ConfigC":
        missing = WIND_FEATURES.difference(set(features))
        if missing:
            raise ValueError(
                f"ConfigC debe contener features de viento. Faltan: {sorted(missing)}"
            )

    config["Features"] = features
    config["InputChannels"] = int(input_channels)

    return config


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def build_sample_table_lookup(sample_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    sample_table = ensure_sample_id_column(sample_table)

    lookup = {}

    for _, row in sample_table.iterrows():
        sid = as_str_id(row["SampleId"])
        lookup[sid] = row.to_dict()

    return lookup


def get_sample_metadata(
    *,
    sample: dict[str, Any],
    split_row: pd.Series | dict[str, Any] | None,
    sample_table_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Construye metadata final combinando:
    1. Fila original en SampleTable.
    2. Fila del split FeatureReady.
    3. Sample["Metadata"].

    El orden importa: valores de Sample["Metadata"] prevalecen.
    """

    metadata: dict[str, Any] = {}

    if sample_table_row is not None:
        metadata.update(sample_table_row)

    if split_row is not None:
        if isinstance(split_row, pd.Series):
            metadata.update(split_row.to_dict())
        elif isinstance(split_row, dict):
            metadata.update(split_row)

    sample_metadata = sample.get("Metadata")
    if isinstance(sample_metadata, dict):
        metadata.update(sample_metadata)

    # Alias útiles por compatibilidad.
    if "id" in metadata and "SampleId" not in metadata:
        metadata["SampleId"] = metadata["id"]

    return metadata


def validate_config_c_metadata(metadata: dict[str, Any], sample_id: str) -> None:
    required = ["meteo:wind_u", "meteo:wind_v"]

    missing = [key for key in required if key not in metadata or pd.isna(metadata[key])]

    if missing:
        preview = list(metadata.keys())[:80]
        raise KeyError(
            f"ConfigC requiere metadatos de viento para SampleId={sample_id}. "
            f"Faltan: {missing}. Claves disponibles: {preview}"
        )


# ---------------------------------------------------------------------
# Split y sample extraction
# ---------------------------------------------------------------------

def read_feature_ready_split(tables_root: Path, split: str) -> pd.DataFrame:
    path = tables_root / FEATURE_READY_FILES[split]

    if not path.exists():
        raise FileNotFoundError(f"No existe split FeatureReady: {path}")

    df = pd.read_csv(path)
    df = ensure_sample_id_column(df)

    return df


def get_sample_array(sample: dict[str, Any], key: str, aliases: list[str]) -> Any:
    if key in sample:
        return sample[key]

    for alias in aliases:
        if alias in sample:
            return sample[alias]

    raise KeyError(
        f"No encontré '{key}' ni aliases {aliases} en la muestra. "
        f"Claves disponibles: {list(sample.keys())}"
    )


def plume_to_mask(plume: Any, expected_hw: tuple[int, int]) -> np.ndarray:
    y = np.asarray(plume)

    if y.ndim == 3:
        if y.shape[0] == 1:
            y = y[0]
        elif y.shape[-1] == 1:
            y = y[..., 0]
        else:
            raise ValueError(f"Plume 3D no esperada: {y.shape}")

    if y.ndim != 2:
        raise ValueError(f"Plume debe ser 2D. Recibido: {y.shape}")

    if tuple(y.shape) != tuple(expected_hw):
        raise ValueError(
            f"Shape de Plume {y.shape} no coincide con feature H,W={expected_hw}"
        )

    return (y > 0).astype(np.uint8)[None, :, :]


def sanitize_features(
    x: np.ndarray,
    *,
    clip_value: float | None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if clip_value is not None:
        clip_value = float(clip_value)
        x = np.clip(x, -clip_value, clip_value).astype(np.float32)

    return x


def call_build_feature_stack(
    *,
    target: Any,
    reference: Any,
    feature_names: list[str],
    feature_config_name: str,
    plume_mask: Any,
    metadata: dict[str, Any],
) -> np.ndarray:
    """
    Llama BuildFeatureStack de forma robusta.
    """

    kwargs = {
        "Target": target,
        "Reference": reference,
        "FeatureNames": feature_names,
        "FeatureConfig": feature_config_name,
        "PlumeMask": plume_mask,
        "ContextMetadata": metadata,
        "Metadata": metadata,
    }

    signature = inspect.signature(BuildFeatureStack)
    accepted = set(signature.parameters.keys())

    filtered_kwargs = {}

    for key, value in kwargs.items():
        if key in accepted:
            filtered_kwargs[key] = value

    # Si BuildFeatureStack acepta **kwargs, podemos pasar todo.
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )

    if accepts_var_kwargs:
        filtered_kwargs = kwargs

    try:
        x = BuildFeatureStack(**filtered_kwargs)
    except TypeError:
        # Fallback para versiones antiguas con firma posicional.
        x = BuildFeatureStack(
            target,
            reference,
            feature_names,
            feature_config_name,
            plume_mask,
        )

    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------
# Construcción de tensores
# ---------------------------------------------------------------------

def build_split(
    *,
    dataset: Any,
    sample_table: pd.DataFrame,
    sample_lookup: dict[str, dict[str, Any]],
    split_table: pd.DataFrame,
    split: str,
    feature_config_name: str,
    feature_names: list[str],
    max_samples_per_split: int | None,
    clip_value: float | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:

    if max_samples_per_split is not None:
        split_table = split_table.head(int(max_samples_per_split)).copy()

    if len(split_table) == 0:
        raise ValueError(f"Split {split} está vacío.")

    x_items: list[np.ndarray] = []
    y_items: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    expected_shape: tuple[int, int, int] | None = None

    print("")
    print(f"--- Building split {split} ---")
    print(f"Samples: {len(split_table)}")
    print(f"FeatureConfig: {feature_config_name}")
    print(f"Features: {feature_names}")

    for order, row in split_table.reset_index(drop=True).iterrows():
        sample_id = as_str_id(row["SampleId"])

        sample = read_sample_flexible(
            dataset=dataset,
            sample_table=sample_table,
            sample_id=sample_id,
        )

        sample_table_row = sample_lookup.get(sample_id)

        metadata = get_sample_metadata(
            sample=sample,
            split_row=row,
            sample_table_row=sample_table_row,
        )

        if feature_config_name == "ConfigC":
            validate_config_c_metadata(metadata, sample_id)

        target = get_sample_array(sample, "Target", ["target"])
        reference = get_sample_array(sample, "Reference", ["reference"])
        plume = get_sample_array(sample, "Plume", ["plume", "Mask", "mask", "GroundTruth"])

        x = call_build_feature_stack(
            target=target,
            reference=reference,
            feature_names=feature_names,
            feature_config_name=feature_config_name,
            plume_mask=plume,
            metadata=metadata,
        )

        if x.ndim != 3:
            raise ValueError(
                f"BuildFeatureStack debe devolver C,H,W. "
                f"SampleId={sample_id}, shape={x.shape}"
            )

        x = sanitize_features(x, clip_value=clip_value)

        if expected_shape is None:
            expected_shape = tuple(x.shape)
            print(f"First tensor shape: {expected_shape}")
        elif tuple(x.shape) != expected_shape:
            raise ValueError(
                f"Shape inconsistente en {split}, SampleId={sample_id}. "
                f"Esperado {expected_shape}, recibido {x.shape}"
            )

        y = plume_to_mask(plume, expected_hw=tuple(x.shape[1:]))

        x_items.append(x.astype(np.float32))
        y_items.append(y.astype(np.uint8))

        wind_speed = np.nan
        wind_cos = np.nan
        wind_sin = np.nan

        if feature_config_name == "ConfigC":
            wind_speed = float(metadata["meteo:wind_u"]) ** 2 + float(metadata["meteo:wind_v"]) ** 2
            wind_speed = float(np.sqrt(wind_speed))
            wind_cos = float(metadata["meteo:wind_u"]) / (wind_speed + 1e-6)
            wind_sin = float(metadata["meteo:wind_v"]) / (wind_speed + 1e-6)

        rows.append(
            {
                "Split": split,
                "Order": int(order),
                "SampleId": sample_id,
                "FeatureConfig": feature_config_name,
                "Channels": int(x.shape[0]),
                "Height": int(x.shape[1]),
                "Width": int(x.shape[2]),
                "MaskPositivePixels": int(y.sum()),
                "FeatureMin": float(np.nanmin(x)),
                "FeatureMax": float(np.nanmax(x)),
                "FeatureMean": float(np.nanmean(x)),
                "FeatureStd": float(np.nanstd(x)),
                "FeatureFinite": bool(np.isfinite(x).all()),
                "WindSource": metadata.get("detection:wind_source", None),
                "WindU": metadata.get("meteo:wind_u", np.nan),
                "WindV": metadata.get("meteo:wind_v", np.nan),
                "WindSpeed10m_Metadata": wind_speed,
                "WindDirCos10m_Metadata": wind_cos,
                "WindDirSin10m_Metadata": wind_sin,
            }
        )

        if (order + 1) % 100 == 0 or (order + 1) == len(split_table):
            print(f"{split}: {order + 1}/{len(split_table)}")

    x_array = np.stack(x_items, axis=0).astype(np.float32)
    y_array = np.stack(y_items, axis=0).astype(np.uint8)

    return x_array, y_array, rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye tensores limpios para ConfigB o ConfigC."
    )

    parser.add_argument(
        "--RunTag",
        required=True,
        help="Identificador del experimento. Ejemplo: Exp271431.",
    )

    parser.add_argument(
        "--ProjectConfig",
        default="Configs/ProjectConfig.yaml",
        help="Ruta a ProjectConfig.yaml.",
    )

    parser.add_argument(
        "--FeatureConfig",
        default="ConfigB",
        choices=["ConfigB", "ConfigC"],
        help="Configuración de features a construir.",
    )

    parser.add_argument(
        "--UseFeatureReadySplits",
        action="store_true",
        help="Compatibilidad CLI. Este script siempre usa splits FeatureReady.",
    )

    parser.add_argument(
        "--MaxSamplesPerSplit",
        type=int,
        default=None,
        help="Limita muestras por split para pruebas rápidas.",
    )

    parser.add_argument(
        "--ClipValue",
        type=float,
        default=8.0,
        help="Clipping absoluto de features. Usa valor negativo para desactivar.",
    )

    args = parser.parse_args()

    run_tag = args.RunTag
    feature_config_name = args.FeatureConfig

    clip_value = None if args.ClipValue is not None and args.ClipValue < 0 else args.ClipValue

    project_config_path = find_project_config_path(args.ProjectConfig)
    project_config = load_yaml(project_config_path)

    data_root, dataset_name = resolve_dataset_location(project_config)

    feature_config = load_feature_config(feature_config_name)
    feature_names = feature_config["Features"]
    input_channels = int(feature_config["InputChannels"])

    if input_channels != len(feature_names):
        raise ValueError(
            f"{feature_config_name}: InputChannels={input_channels}, "
            f"pero len(Features)={len(feature_names)}"
        )

    run_root = PROJECT_ROOT / "Outputs" / "Experiments" / run_tag
    root_tables_dir = run_root / "Tables"

    feature_root = run_root / feature_config_name
    feature_dir = feature_root / "Features"
    tables_dir = feature_root / "Tables"
    audit_dir = feature_root / "Audit"

    feature_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("=== STEP07 CLEAN FEATURE BUILD START ===")
    print(f"ProjectRoot: {PROJECT_ROOT}")
    print(f"ProjectConfig: {project_config_path}")
    print(f"RunTag: {run_tag}")
    print(f"FeatureConfig: {feature_config_name}")
    print(f"InputChannels: {input_channels}")
    print(f"FeatureNames: {feature_names}")
    print(f"DataRoot: {data_root}")
    print(f"DatasetName: {dataset_name}")
    print(f"RootTablesDir: {root_tables_dir}")
    print(f"FeatureDir: {feature_dir}")
    print(f"ClipValue: {clip_value}")

    if not root_tables_dir.exists():
        raise FileNotFoundError(
            f"No existe carpeta de tablas del run: {root_tables_dir}. "
            "Ejecuta primero Step01-Step06."
        )

    dataset, sample_table = load_dataset_flexible(data_root, dataset_name)
    sample_table = ensure_sample_id_column(sample_table)
    sample_lookup = build_sample_table_lookup(sample_table)

    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        split_table = read_feature_ready_split(root_tables_dir, split)

        original_samples = len(split_table)

        x_array, y_array, rows = build_split(
            dataset=dataset,
            sample_table=sample_table,
            sample_lookup=sample_lookup,
            split_table=split_table,
            split=split,
            feature_config_name=feature_config_name,
            feature_names=feature_names,
            max_samples_per_split=args.MaxSamplesPerSplit,
            clip_value=clip_value,
        )

        if x_array.shape[1] != input_channels:
            raise ValueError(
                f"{split}: X tiene {x_array.shape[1]} canales, "
                f"pero {feature_config_name} espera {input_channels}."
            )

        x_path = feature_dir / f"{split}Features.npy"
        y_path = feature_dir / f"{split}Masks.npy"

        np.save(x_path, x_array)
        np.save(y_path, y_array)

        sample_rows.extend(rows)

        summary_rows.append(
            {
                "RunTag": run_tag,
                "FeatureConfig": feature_config_name,
                "Split": split,
                "Samples": int(x_array.shape[0]),
                "OriginalSplitSamples": int(original_samples),
                "Channels": int(x_array.shape[1]),
                "Height": int(x_array.shape[2]),
                "Width": int(x_array.shape[3]),
                "FeaturePath": str(x_path),
                "MaskPath": str(y_path),
                "FeatureName": ", ".join(feature_names),
                "ClipValue": clip_value,
                "AnyNonFinite": bool(not np.isfinite(x_array).all()),
                "MaskPositivePixels": int(y_array.sum()),
            }
        )

        print("")
        print(f"Saved {split}")
        print(f"X: {x_array.shape} -> {x_path}")
        print(f"Y: {y_array.shape} -> {y_path}")

    summary = pd.DataFrame(summary_rows)
    sample_summary = pd.DataFrame(sample_rows)

    summary_path = tables_dir / "FeatureBuildSummary.csv"
    sample_summary_path = tables_dir / "FeatureBuildSampleSummary.csv"
    audit_path = audit_dir / "FeatureBuildAudit.json"

    summary.to_csv(summary_path, index=False)
    sample_summary.to_csv(sample_summary_path, index=False)

    save_json(
        audit_path,
        {
            "RunTag": run_tag,
            "FeatureConfig": feature_config_name,
            "InputChannels": input_channels,
            "FeatureNames": feature_names,
            "ProjectRoot": str(PROJECT_ROOT),
            "ProjectConfig": str(project_config_path),
            "DataRoot": data_root,
            "DatasetName": dataset_name,
            "RootTablesDir": str(root_tables_dir),
            "FeatureDir": str(feature_dir),
            "TablesDir": str(tables_dir),
            "AuditDir": str(audit_dir),
            "SummaryPath": str(summary_path),
            "SampleSummaryPath": str(sample_summary_path),
            "ClipValue": clip_value,
            "MaxSamplesPerSplit": args.MaxSamplesPerSplit,
            "UseFeatureReadySplits": bool(args.UseFeatureReadySplits),
            "Methodology": {
                "UsesGroundTruthForFeatures": False,
                "UsesPlumeMaskForFeatures": False,
                "UsesPlumeMaskOnlyAsTargetY": True,
                "ConfigCWindEncoding": (
                    "WindSpeed10m=sqrt(u^2+v^2), "
                    "WindDirCos10m=u/(speed+eps), "
                    "WindDirSin10m=v/(speed+eps)"
                ),
            },
        },
    )

    print("")
    print("=== STEP07 CLEAN FEATURE BUILD COMPLETED ===")
    print(f"RunTag: {run_tag}")
    print(f"FeatureConfig: {feature_config_name}")
    print(f"Summary: {summary_path}")
    print(f"SampleSummary: {sample_summary_path}")
    print(f"Audit: {audit_path}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
