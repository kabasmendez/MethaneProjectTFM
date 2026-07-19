"""
ColorMaps.py

Centraliza colormaps, normalización robusta y estilos por tipo de producto.

Responsabilidades:
- normalizar RGB y composiciones Sentinel;
- calcular rangos robustos por percentiles;
- centrar productos diferenciales en cero;
- devolver estilos visuales para features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


def ValidateFiniteArray(Array: np.ndarray, Name: str) -> np.ndarray:
    """Valida que un array contenga al menos algún valor finito."""
    Array = np.asarray(Array, dtype=np.float32)

    if Array.size == 0:
        raise ValueError(f"{Name}: array vacío.")

    if not np.isfinite(Array).any():
        raise ValueError(f"{Name}: array sin valores finitos.")

    return Array


def RobustLimits(Array: np.ndarray, LowerPercentile: float, UpperPercentile: float) -> tuple[float, float]:
    """Calcula límites robustos por percentiles usando valores finitos."""
    Array = ValidateFiniteArray(Array, "RobustLimits")
    FiniteValues = Array[np.isfinite(Array)]

    Vmin = float(np.percentile(FiniteValues, LowerPercentile))
    Vmax = float(np.percentile(FiniteValues, UpperPercentile))

    if np.isclose(Vmin, Vmax):
        Vmin = float(np.nanmin(FiniteValues))
        Vmax = float(np.nanmax(FiniteValues))

    if np.isclose(Vmin, Vmax):
        Vmin -= 1.0
        Vmax += 1.0

    return Vmin, Vmax


def BuildNormalize(Array: np.ndarray, StyleConfig: dict[str, Any]) -> Normalize:
    """Construye normalización matplotlib según configuración."""
    if "Vmin" in StyleConfig and "Vmax" in StyleConfig:
        return Normalize(vmin=float(StyleConfig["Vmin"]), vmax=float(StyleConfig["Vmax"]))

    Lower, Upper = StyleConfig.get("RobustPercentiles", [2, 98])
    Vmin, Vmax = RobustLimits(Array, float(Lower), float(Upper))

    if bool(StyleConfig.get("CenterZero", False)):
        Limit = max(abs(Vmin), abs(Vmax))
        if np.isclose(Limit, 0.0):
            Limit = 1.0
        return TwoSlopeNorm(vmin=-Limit, vcenter=0.0, vmax=Limit)

    return Normalize(vmin=Vmin, vmax=Vmax)


def NormalizeImage01(
    Array: np.ndarray,
    LowerPercentile: float = 2,
    UpperPercentile: float = 98,
    Gamma: float = 1.0,
) -> np.ndarray:
    """Normaliza una banda o imagen a rango 0–1 con percentiles robustos."""
    Array = ValidateFiniteArray(Array, "NormalizeImage01")
    Vmin, Vmax = RobustLimits(Array, LowerPercentile, UpperPercentile)

    Normalized = (Array - Vmin) / (Vmax - Vmin)
    Normalized = np.clip(Normalized, 0.0, 1.0)

    if Gamma <= 0:
        raise ValueError(f"Gamma debe ser positivo. Valor recibido: {Gamma}")

    if not np.isclose(Gamma, 1.0):
        Normalized = Normalized ** (1.0 / Gamma)

    return Normalized.astype(np.float32)


def BuildRgbImage(
    Red: np.ndarray,
    Green: np.ndarray,
    Blue: np.ndarray,
    LowerPercentile: float = 2,
    UpperPercentile: float = 98,
    Gamma: float = 1.0,
) -> np.ndarray:
    """Construye imagen RGB normalizada H x W x 3."""
    Red01 = NormalizeImage01(Red, LowerPercentile, UpperPercentile, Gamma)
    Green01 = NormalizeImage01(Green, LowerPercentile, UpperPercentile, Gamma)
    Blue01 = NormalizeImage01(Blue, LowerPercentile, UpperPercentile, Gamma)

    return np.stack([Red01, Green01, Blue01], axis=-1)


def GetFeatureStyle(VisualConfig: dict[str, Any], FeatureName: str) -> tuple[str, dict[str, Any]]:
    """Devuelve nombre de estilo y configuración para una feature."""
    FeatureStyles = VisualConfig["FeatureStyles"]

    if FeatureName not in FeatureStyles:
        raise KeyError(f"No existe estilo definido para la feature: {FeatureName}")

    StyleName = FeatureStyles[FeatureName]["Style"]
    ColorMaps = VisualConfig["ColorMaps"]

    if StyleName not in ColorMaps:
        raise KeyError(f"Feature {FeatureName} usa Style={StyleName}, pero no existe en ColorMaps.")

    return StyleName, ColorMaps[StyleName]
