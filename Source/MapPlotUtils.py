"""
MapPlotUtils.py

Funciones comunes para figuras tipo mapa o panel satelital.

Incluye:
- colorbars estándar;
- anotaciones discretas;
- escala aproximada basada en tamaño de píxel;
- flecha norte simple;
- limpieza de ejes;
- contornos de plumas sin relleno;
- leyendas de plumas.
"""

from __future__ import annotations

from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def FormatImageAxis(Axis, VisualConfig: dict[str, Any]) -> None:
    """Aplica formato estándar a ejes de imagen."""
    ShowAxes = bool(VisualConfig["Visualization"]["MapElements"].get("ShowAxes", False))

    if not ShowAxes:
        Axis.set_xticks([])
        Axis.set_yticks([])
        Axis.set_xlabel("")
        Axis.set_ylabel("")


def AddColorbar(Figure, Axis, ImageHandle, Label: str, VisualConfig: dict[str, Any]) -> None:
    """Añade colorbar estándar a una imagen."""
    FontSizes = VisualConfig["Visualization"]["FontSizes"]
    Colorbar = Figure.colorbar(ImageHandle, ax=Axis, fraction=0.046, pad=0.04)
    Colorbar.set_label(Label, fontsize=FontSizes.get("ColorbarLabel", 9))
    Colorbar.ax.tick_params(labelsize=FontSizes.get("ColorbarTick", 8))


def AddNorthArrow(Axis, VisualConfig: dict[str, Any]) -> None:
    """Añade flecha norte simple si está activada."""
    MapElements = VisualConfig["Visualization"]["MapElements"]

    if not bool(MapElements.get("ShowNorthArrow", True)):
        return

    Axis.annotate(
        "N",
        xy=(0.92, 0.82),
        xytext=(0.92, 0.64),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", linewidth=1.0, color="black"),
        ha="center",
        va="center",
        fontsize=VisualConfig["Visualization"]["FontSizes"].get("Annotation", 8),
        color="black",
    )


def AddScaleText(Axis, VisualConfig: dict[str, Any]) -> None:
    """Añade texto de escala aproximada si está activado."""
    MapElements = VisualConfig["Visualization"]["MapElements"]

    if not bool(MapElements.get("ShowScaleBar", True)):
        return

    Text = MapElements.get("ApproxExtentText", "")
    if not Text:
        PixelSize = MapElements.get("PixelSizeMeters", 20)
        TileSize = MapElements.get("TileSizePixels", 200)
        ExtentKm = PixelSize * TileSize / 1000.0
        Text = (
            f"{TileSize} x {TileSize} px - {PixelSize} m/pixel - "
            f"approx. {ExtentKm:.1f} km x {ExtentKm:.1f} km"
        )

    Axis.text(
        0.02,
        0.03,
        Text,
        transform=Axis.transAxes,
        fontsize=VisualConfig["Visualization"]["FontSizes"].get("Annotation", 8),
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
    )


def AddMetadataText(Axis, Lines: list[str], VisualConfig: dict[str, Any]) -> None:
    """Añade metadatos de muestra dentro del panel."""
    if not Lines:
        return

    Axis.text(
        0.02,
        0.98,
        "\n".join(Lines),
        transform=Axis.transAxes,
        fontsize=VisualConfig["Visualization"]["FontSizes"].get("Annotation", 8),
        color="black",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=3),
    )


def GetPlumeContourStyle(VisualConfig: dict[str, Any], PlumeRole: str) -> dict[str, Any]:
    """Devuelve estilo de contorno para un rol de pluma."""
    Contours = VisualConfig.get("PlumeContours", {})

    if PlumeRole not in Contours:
        raise KeyError(
            f"No existe estilo de contorno para PlumeRole={PlumeRole}. "
            f"Roles disponibles: {list(Contours.keys())}"
        )

    return Contours[PlumeRole]


def DrawPlumeContour(
    Axis,
    Mask: np.ndarray,
    VisualConfig: dict[str, Any],
    PlumeRole: str = "GroundTruth",
    Label: str | None = None,
) -> None:
    """
    Dibuja contorno de pluma sin relleno.

    Regla oficial:
    - Las plumas ground truth y predichas se dibujan como contorno.
    - No se rellenan en visualizaciones estándar.
    """
    if Mask is None:
        return

    BinaryMask = np.asarray(Mask > 0).astype(float)

    if BinaryMask.ndim != 2:
        raise ValueError(f"Mask debe ser 2D. Forma recibida: {BinaryMask.shape}")

    if not np.any(BinaryMask > 0):
        return

    Style = GetPlumeContourStyle(VisualConfig, PlumeRole)

    Axis.contour(
        BinaryMask,
        levels=[0.5],
        colors=Style.get("Color", "white"),
        linewidths=float(Style.get("LineWidth", 1.5)),
        linestyles=Style.get("LineStyle", "solid"),
    )


def AddPlumeLegend(
    Figure,
    VisualConfig: dict[str, Any],
    PlumeRoles: list[str],
    Location: str | None = None,
) -> None:
    """Añade leyenda general de contornos de plumas."""
    UniqueRoles = []
    for Role in PlumeRoles:
        if Role not in UniqueRoles:
            UniqueRoles.append(Role)

    if not UniqueRoles:
        return

    LegendConfig = VisualConfig.get("Legends", {})
    if not bool(LegendConfig.get("ShowPlumeLegend", True)):
        return

    Handles = []

    for Role in UniqueRoles:
        Style = GetPlumeContourStyle(VisualConfig, Role)
        Handles.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor=Style.get("Color", "white"),
                linewidth=float(Style.get("LineWidth", 1.5)),
                label=Style.get("Label", Role),
            )
        )

    Figure.legend(
        handles=Handles,
        loc=Location or LegendConfig.get("Location", "lower center"),
        ncol=max(1, len(Handles)),
        frameon=True,
        framealpha=float(LegendConfig.get("FrameAlpha", 0.90)),
        fontsize=int(LegendConfig.get("FontSize", 8)),
    )


def BuildErrorMap(GroundTruthMask: np.ndarray, PredictionMask: np.ndarray) -> np.ndarray:
    """
    Construye mapa de error codificado:
    0 = transparente / TN
    1 = TP
    2 = FP
    3 = FN
    """
    GroundTruth = np.asarray(GroundTruthMask).astype(bool)
    Prediction = np.asarray(PredictionMask).astype(bool)

    if GroundTruth.shape != Prediction.shape:
        raise ValueError(
            f"GroundTruth y Prediction deben tener la misma forma. "
            f"GT={GroundTruth.shape}, Pred={Prediction.shape}"
        )

    ErrorMap = np.zeros(GroundTruth.shape, dtype=np.uint8)
    ErrorMap[GroundTruth & Prediction] = 1
    ErrorMap[~GroundTruth & Prediction] = 2
    ErrorMap[GroundTruth & ~Prediction] = 3

    return ErrorMap
