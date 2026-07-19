"""
SatelliteVisualization.py

Funciones de visualización para Sentinel-2, features, máscaras y predicciones.

Este módulo no decide rutas ni nombres de salida. Recibe datos, estilo y Axis/Figure.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from Source.ColorMaps import BuildNormalize, BuildRgbImage, GetFeatureStyle
from Source.MapPlotUtils import (
    AddColorbar,
    AddMetadataText,
    AddNorthArrow,
    AddScaleText,
    BuildErrorMap,
    FormatImageAxis,
)


def PlotSingleBand(
    Axis,
    Figure,
    Array: np.ndarray,
    Title: str,
    StyleConfig: dict[str, Any],
    VisualConfig: dict[str, Any],
) -> None:
    """Dibuja banda o feature 2D con colormap y colorbar estándar."""
    Normalize = BuildNormalize(Array, StyleConfig)
    Cmap = StyleConfig.get("Cmap", "viridis")
    ImageHandle = Axis.imshow(
        Array,
        cmap=Cmap,
        norm=Normalize,
        interpolation=VisualConfig["Visualization"]["Image"].get("Interpolation", "nearest"),
        origin=VisualConfig["Visualization"]["Image"].get("Origin", "upper"),
    )

    Axis.set_title(Title)
    FormatImageAxis(Axis, VisualConfig)

    if bool(StyleConfig.get("Colorbar", True)):
        AddColorbar(
            Figure,
            Axis,
            ImageHandle,
            StyleConfig.get("ColorbarLabel", ""),
            VisualConfig,
        )


def PlotFeature(
    Axis,
    Figure,
    Array: np.ndarray,
    FeatureName: str,
    VisualConfig: dict[str, Any],
) -> None:
    """Dibuja una feature usando su estilo definido en VisualizationConfig."""
    _, StyleConfig = GetFeatureStyle(VisualConfig, FeatureName)
    Title = VisualConfig["FeatureStyles"][FeatureName].get("Title", FeatureName)
    PlotSingleBand(Axis, Figure, Array, Title, StyleConfig, VisualConfig)


def PlotRgbComposite(
    Axis,
    Red: np.ndarray,
    Green: np.ndarray,
    Blue: np.ndarray,
    Title: str,
    VisualConfig: dict[str, Any],
    CompositeName: str = "RGB",
) -> None:
    """Dibuja composición RGB con stretch robusto."""
    CompositeConfig = VisualConfig["ColorMaps"][CompositeName]
    Lower, Upper = CompositeConfig.get("RobustPercentiles", [2, 98])
    Gamma = float(CompositeConfig.get("Gamma", 1.0))

    RgbImage = BuildRgbImage(
        Red,
        Green,
        Blue,
        LowerPercentile=float(Lower),
        UpperPercentile=float(Upper),
        Gamma=Gamma,
    )

    Axis.imshow(
        RgbImage,
        interpolation=VisualConfig["Visualization"]["Image"].get("Interpolation", "nearest"),
        origin=VisualConfig["Visualization"]["Image"].get("Origin", "upper"),
    )
    Axis.set_title(Title)
    FormatImageAxis(Axis, VisualConfig)
    AddNorthArrow(Axis, VisualConfig)
    AddScaleText(Axis, VisualConfig)


def PlotMaskOverlay(
    Axis,
    BaseImage: np.ndarray,
    Mask: np.ndarray,
    Title: str,
    Color: str,
    Alpha: float,
    VisualConfig: dict[str, Any],
    MetadataLines: list[str] | None = None,
) -> None:
    """Dibuja máscara binaria sobre imagen base RGB."""
    Axis.imshow(BaseImage)
    Masked = np.ma.masked_where(~np.asarray(Mask).astype(bool), Mask)
    Axis.imshow(Masked, cmap=ListedColormap([Color]), alpha=Alpha)
    Axis.set_title(Title)
    FormatImageAxis(Axis, VisualConfig)

    if MetadataLines:
        AddMetadataText(Axis, MetadataLines, VisualConfig)


def PlotProbability(
    Axis,
    Figure,
    Probability: np.ndarray,
    VisualConfig: dict[str, Any],
    Title: str = "Probability",
) -> None:
    """Dibuja mapa de probabilidad 0–1 con viridis."""
    StyleConfig = VisualConfig["ColorMaps"]["Probability"]
    PlotSingleBand(Axis, Figure, Probability, Title, StyleConfig, VisualConfig)


def PlotErrorMap(
    Axis,
    GroundTruthMask: np.ndarray,
    PredictionMask: np.ndarray,
    VisualConfig: dict[str, Any],
    Title: str = "Error map",
) -> None:
    """Dibuja mapa TP/FP/FN."""
    ErrorMap = BuildErrorMap(GroundTruthMask, PredictionMask)
    ErrorConfig = VisualConfig["ColorMaps"]["ErrorMap"]

    Colors = [
        (0, 0, 0, 0),
        ErrorConfig["TruePositiveColor"],
        ErrorConfig["FalsePositiveColor"],
        ErrorConfig["FalseNegativeColor"],
    ]

    Cmap = ListedColormap(Colors)
    Norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], Cmap.N)

    Axis.imshow(ErrorMap, cmap=Cmap, norm=Norm)
    Axis.set_title(Title)
    FormatImageAxis(Axis, VisualConfig)

    if bool(ErrorConfig.get("Legend", True)):
        Axis.text(
            0.02,
            0.03,
            "TP yellow · FP red · FN blue",
            transform=Axis.transAxes,
            fontsize=VisualConfig["Visualization"]["FontSizes"].get("Annotation", 8),
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        )
