"""
ContextFeatures.py

Inspección y cálculo de variables contextuales para ConfigC.

Variables previstas:
- WindSin
- WindCos
- WindSpeed
- SolarAzimuthSin
- SolarAzimuthCos

No se asume que las columnas existan. Primero se inspeccionan metadatos.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


WIND_U_CANDIDATES = ["meteo:wind_u", "wind_u", "WindU", "u10", "WindEast"]
WIND_V_CANDIDATES = ["meteo:wind_v", "wind_v", "WindV", "v10", "WindNorth"]

SOLAR_AZIMUTH_CANDIDATES = [
    "satellite:saa",
    "satellite:solar_azimuth",
    "solar_azimuth",
    "SolarAzimuth",
]

SOLAR_ZENITH_CANDIDATES = [
    "satellite:sza",
    "satellite:solar_zenith",
    "solar_zenith",
    "SolarZenith",
]


def FindFirstExistingColumn(Columns: Iterable[str], Candidates: list[str]) -> str | None:
    """Devuelve la primera columna candidata encontrada."""
    ColumnSet = set(Columns)

    for Candidate in Candidates:
        if Candidate in ColumnSet:
            return Candidate

    return None


def FindContextColumnCandidates(DataFrame: pd.DataFrame) -> dict[str, str | None]:
    """Busca columnas candidatas para viento y geometría solar."""
    Columns = list(DataFrame.columns)

    return {
        "WindU": FindFirstExistingColumn(Columns, WIND_U_CANDIDATES),
        "WindV": FindFirstExistingColumn(Columns, WIND_V_CANDIDATES),
        "SolarAzimuth": FindFirstExistingColumn(Columns, SOLAR_AZIMUTH_CANDIDATES),
        "SolarZenith": FindFirstExistingColumn(Columns, SOLAR_ZENITH_CANDIDATES),
    }


def ComputeWindPolar(WindU: float, WindV: float) -> dict[str, float]:
    """Calcula WindSin, WindCos y WindSpeed desde componentes U/V."""
    WindU = float(WindU)
    WindV = float(WindV)

    if not math.isfinite(WindU) or not math.isfinite(WindV):
        raise ValueError(f"Componentes de viento no finitas: WindU={WindU}, WindV={WindV}")

    WindSpeed = math.sqrt(WindU * WindU + WindV * WindV)

    if WindSpeed <= 0:
        return {"WindSin": 0.0, "WindCos": 0.0, "WindSpeed": 0.0}

    return {
        "WindSin": float(WindV / WindSpeed),
        "WindCos": float(WindU / WindSpeed),
        "WindSpeed": float(WindSpeed),
    }


def ComputeAngleSinCosDegrees(AngleDegrees: float) -> dict[str, float]:
    """Convierte ángulo en grados a seno/coseno."""
    AngleDegrees = float(AngleDegrees)

    if not math.isfinite(AngleDegrees):
        raise ValueError(f"Ángulo no finito: {AngleDegrees}")

    AngleRadians = math.radians(AngleDegrees)

    return {
        "Sin": float(math.sin(AngleRadians)),
        "Cos": float(math.cos(AngleRadians)),
    }


def ComputeSolarAzimuthPolar(SolarAzimuthDegrees: float) -> dict[str, float]:
    """Calcula SolarAzimuthSin y SolarAzimuthCos."""
    Values = ComputeAngleSinCosDegrees(SolarAzimuthDegrees)

    return {
        "SolarAzimuthSin": Values["Sin"],
        "SolarAzimuthCos": Values["Cos"],
    }


def ExpandScalarToImage(Value: float, Height: int, Width: int) -> np.ndarray:
    """Expande un valor escalar a imagen H x W."""
    if not math.isfinite(float(Value)):
        raise ValueError(f"No se puede expandir un valor no finito: {Value}")

    return np.full((Height, Width), float(Value), dtype=np.float32)


def BuildContextSummary(DataFrame: pd.DataFrame) -> pd.DataFrame:
    """Construye tabla de disponibilidad de metadatos contextuales."""
    Candidates = FindContextColumnCandidates(DataFrame)
    Rows = []

    for Concept, Column in Candidates.items():
        if Column is None:
            Rows.append(
                {
                    "Concept": Concept,
                    "Column": "",
                    "Exists": False,
                    "NonNull": 0,
                    "Null": len(DataFrame),
                    "Numeric": False,
                    "Min": None,
                    "Max": None,
                    "Mean": None,
                }
            )
            continue

        Series = DataFrame[Column]
        NumericSeries = pd.to_numeric(Series, errors="coerce")

        Rows.append(
            {
                "Concept": Concept,
                "Column": Column,
                "Exists": True,
                "NonNull": int(Series.notna().sum()),
                "Null": int(Series.isna().sum()),
                "Numeric": bool(NumericSeries.notna().sum() > 0),
                "Min": float(NumericSeries.min()) if NumericSeries.notna().any() else None,
                "Max": float(NumericSeries.max()) if NumericSeries.notna().any() else None,
                "Mean": float(NumericSeries.mean()) if NumericSeries.notna().any() else None,
            }
        )

    return pd.DataFrame(Rows)
