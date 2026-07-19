"""
ReadTacoSample.py

Lectura robusta de muestras TACO para MethaneProjectTFM.

Responsabilidades:
- localizar una muestra por SampleId;
- leer productos raster target, reference, plume, ch4 y dem cuando existan;
- validar formas, tipos y valores finitos;
- devolver un diccionario limpio para visualización y futuras fases.

Este módulo no genera figuras ni features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PRODUCTS = ("target", "reference", "plume")
OPTIONAL_PRODUCTS = ("ch4", "dem")


def FindSampleIndexById(SampleTable: pd.DataFrame, SampleId: str) -> int:
    """Busca la posición entera de una muestra en Dataset.data."""
    CandidateColumns = ["id", "SampleId", "sample_id"]
    IdColumn = None

    for Column in CandidateColumns:
        if Column in SampleTable.columns:
            IdColumn = Column
            break

    if IdColumn is None:
        raise KeyError(f"No se encontró columna de ID en SampleTable. Columnas: {list(SampleTable.columns)}")

    Matches = SampleTable.index[SampleTable[IdColumn].astype(str) == str(SampleId)].tolist()

    if len(Matches) == 0:
        raise ValueError(f"No se encontró SampleId en SampleTable: {SampleId}")

    if len(Matches) > 1:
        raise ValueError(f"SampleId duplicado en SampleTable: {SampleId}. Coincidencias: {len(Matches)}")

    return int(Matches[0])


def ReadTacoSampleByIndex(Dataset: Any, SampleIndex: int) -> Any:
    """
    Lee una muestra TACO por índice entero.

    Se espera que Dataset.data tenga método read(Index).
    """
    if not hasattr(Dataset, "data"):
        raise AttributeError("El objeto Dataset no tiene atributo .data")

    if not hasattr(Dataset.data, "read"):
        raise AttributeError("Dataset.data no tiene método read(Index).")

    return Dataset.data.read(SampleIndex)


def ReadRasterFromSample(
    Sample: Any,
    ProductName: str,
    ReadAllBands: bool,
    Required: bool = True,
) -> tuple[np.ndarray | None, dict[str, Any] | None, str | None]:
    """
    Lee un raster de una muestra usando Sample.read(ProductName) + rasterio.

    Devuelve:
    - Array numpy float32;
    - profile rasterio;
    - ruta o VSI path devuelta por tacoreader.
    """
    try:
        import rasterio
    except ImportError as Error:
        raise ImportError(
            "No se pudo importar rasterio. Instálalo en el entorno deep con: "
            "python -m pip install rasterio"
        ) from Error

    if not hasattr(Sample, "read"):
        raise AttributeError("El objeto Sample no tiene método read(ProductName).")

    try:
        AssetPath = Sample.read(ProductName)
    except Exception as Error:
        if Required:
            raise RuntimeError(f"No se pudo leer el producto requerido '{ProductName}' desde Sample.") from Error
        return None, None, None

    try:
        with rasterio.open(AssetPath) as Raster:
            if ReadAllBands:
                Array = Raster.read().astype(np.float32)
            else:
                Array = Raster.read(1).astype(np.float32)

            Profile = Raster.profile.copy()
    except Exception as Error:
        if Required:
            raise RuntimeError(
                f"No se pudo abrir con rasterio el producto '{ProductName}'. AssetPath={AssetPath}"
            ) from Error
        return None, None, str(AssetPath)

    return Array, Profile, str(AssetPath)


def ValidateArrayFinite(Array: np.ndarray, ProductName: str) -> None:
    """Valida que un array tenga valores finitos."""
    if Array.size == 0:
        raise ValueError(f"{ProductName}: array vacío.")

    if not np.isfinite(Array).any():
        raise ValueError(f"{ProductName}: no contiene valores finitos.")


def ValidateProductShapes(
    SampleData: dict[str, Any],
    ExpectedShapes: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Valida formas principales de una muestra."""
    Target = SampleData["Target"]
    Reference = SampleData["Reference"]
    Plume = SampleData["Plume"]

    ValidateArrayFinite(Target, "Target")
    ValidateArrayFinite(Reference, "Reference")
    ValidateArrayFinite(Plume, "Plume")

    Validation = {
        "SampleId": SampleData["SampleId"],
        "TargetShape": list(Target.shape),
        "ReferenceShape": list(Reference.shape),
        "PlumeShape": list(Plume.shape),
        "CH4Shape": list(SampleData["CH4"].shape) if SampleData.get("CH4") is not None else None,
        "DemShape": list(SampleData["Dem"].shape) if SampleData.get("Dem") is not None else None,
        "TargetFinite": bool(np.isfinite(Target).any()),
        "ReferenceFinite": bool(np.isfinite(Reference).any()),
        "PlumeFinite": bool(np.isfinite(Plume).any()),
        "CH4Finite": bool(np.isfinite(SampleData["CH4"]).any()) if SampleData.get("CH4") is not None else False,
        "DemFinite": bool(np.isfinite(SampleData["Dem"]).any()) if SampleData.get("Dem") is not None else False,
        "PlumePixels": int(np.nansum(Plume > 0)),
        "IsValid": True,
    }

    if ExpectedShapes:
        ShapeMap = {
            "Target": Validation["TargetShape"],
            "Reference": Validation["ReferenceShape"],
            "Plume": Validation["PlumeShape"],
            "CH4": Validation["CH4Shape"],
        }

        for ProductName, ExpectedShape in ExpectedShapes.items():
            if ProductName not in ShapeMap:
                continue

            ObservedShape = ShapeMap[ProductName]
            if ObservedShape is None:
                continue

            if list(ExpectedShape) != list(ObservedShape):
                Validation["IsValid"] = False
                Validation[f"{ProductName}ShapeError"] = {
                    "Expected": list(ExpectedShape),
                    "Observed": ObservedShape,
                }

    return Validation


def ReadFullTacoSample(
    Dataset: Any,
    SampleTable: pd.DataFrame,
    SampleId: str,
    ExpectedShapes: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """
    Lee productos principales de una muestra.

    Productos requeridos:
    - target
    - reference
    - plume

    Productos opcionales:
    - ch4
    - dem
    """
    SampleIndex = FindSampleIndexById(SampleTable, SampleId)
    Sample = ReadTacoSampleByIndex(Dataset, SampleIndex)

    Metadata = SampleTable.loc[SampleIndex].to_dict()

    Target, TargetProfile, TargetPath = ReadRasterFromSample(
        Sample,
        "target",
        ReadAllBands=True,
        Required=True,
    )

    Reference, ReferenceProfile, ReferencePath = ReadRasterFromSample(
        Sample,
        "reference",
        ReadAllBands=True,
        Required=True,
    )

    Plume, PlumeProfile, PlumePath = ReadRasterFromSample(
        Sample,
        "plume",
        ReadAllBands=False,
        Required=True,
    )

    CH4, CH4Profile, CH4Path = ReadRasterFromSample(
        Sample,
        "ch4",
        ReadAllBands=False,
        Required=False,
    )

    Dem, DemProfile, DemPath = ReadRasterFromSample(
        Sample,
        "dem",
        ReadAllBands=False,
        Required=False,
    )

    SampleData = {
        "SampleId": str(SampleId),
        "SampleIndex": int(SampleIndex),
        "Metadata": Metadata,
        "Target": Target,
        "Reference": Reference,
        "Plume": Plume,
        "CH4": CH4,
        "Dem": Dem,
        "Profiles": {
            "Target": TargetProfile,
            "Reference": ReferenceProfile,
            "Plume": PlumeProfile,
            "CH4": CH4Profile,
            "Dem": DemProfile,
        },
        "Paths": {
            "Target": TargetPath,
            "Reference": ReferencePath,
            "Plume": PlumePath,
            "CH4": CH4Path,
            "Dem": DemPath,
        },
    }

    SampleData["Validation"] = ValidateProductShapes(SampleData, ExpectedShapes=ExpectedShapes)

    return SampleData
