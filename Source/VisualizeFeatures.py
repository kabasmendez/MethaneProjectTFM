"""
VisualizeFeatures.py

Preview visual de features espectrales.

Reglas:
- La pluma ground truth se dibuja solo como contorno.
- No se rellena la pluma en paneles RGB ni en canales.
- La leyenda se toma del estándar visual global.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from Source.ColorMaps import BuildNormalize
from Source.MapPlotUtils import AddColorbar, AddPlumeLegend, DrawPlumeContour, FormatImageAxis
from Source.VisualizeSamples import BuildRgbFromSentinel, BuildSwirComposite


def BuildFeaturePanelSpecs(FeatureConfig: str, IncludeCH4: bool = True) -> list[dict[str, Any]]:
    """Define paneles visuales para ConfigA o ConfigB."""
    if FeatureConfig not in ["ConfigA", "ConfigB"]:
        raise ValueError(f"FeatureConfig no soportada: {FeatureConfig}")

    Panels: list[dict[str, Any]] = [
        {"Kind": "rgb", "ImageKey": "TargetRGB", "Title": "Target RGB"},
        {"Kind": "rgb", "ImageKey": "ReferenceRGB", "Title": "Reference RGB"},
        {"Kind": "rgb", "ImageKey": "TargetSWIR", "Title": "Target SWIR B12-B11-B8A"},
        {"Kind": "rgb", "ImageKey": "ReferenceSWIR", "Title": "Reference SWIR B12-B11-B8A"},
    ]

    if IncludeCH4:
        Panels.append(
            {
                "Kind": "channel",
                "ImageKey": "CH4",
                "Title": "CH4 enhancement",
                "Cmap": "plasma",
                "CenterZero": False,
                "ColorbarLabel": "Delta XCH4",
            }
        )

    FeaturePanels = [
        ("B8A", "B8A", "gray", False, "Reflectance"),
        ("B11", "B11 SWIR1", "magma", False, "Reflectance"),
        ("B12", "B12 SWIR2", "magma", False, "Reflectance"),
        ("NDSWIR", "NDSWIR", "coolwarm", True, "Feature value"),
        ("RatioB12B11", "Ratio B12/B11", "plasma", False, "Ratio value"),
        ("RatioB12B8A", "Ratio B12/B8A", "plasma", False, "Ratio value"),
        ("MBMP", "MBMP classic", "inferno", False, "Feature value"),
    ]

    if FeatureConfig == "ConfigB":
        FeaturePanels.extend(
            [
                ("MBMPPlus", "MBMPPlus", "viridis", False, "Feature value"),
                (
                    "DualEnhancementB12B11",
                    "DualEnhancement B12/B11",
                    "coolwarm",
                    True,
                    "Feature value",
                ),
            ]
        )

    for ImageKey, Title, Cmap, CenterZero, ColorbarLabel in FeaturePanels:
        Panels.append(
            {
                "Kind": "channel",
                "ImageKey": ImageKey,
                "Title": Title,
                "Cmap": Cmap,
                "CenterZero": CenterZero,
                "ColorbarLabel": ColorbarLabel,
            }
        )

    Panels.append(
        {
            "Kind": "mask",
            "ImageKey": "Plume",
            "Title": "Plume ground truth",
            "Cmap": "gray",
        }
    )

    return Panels


def PlotChannel(
    Figure,
    Axis,
    Image: np.ndarray,
    Plume: np.ndarray,
    Title: str,
    Cmap: str,
    CenterZero: bool,
    ColorbarLabel: str,
    VisualConfig: dict[str, Any],
) -> None:
    """Dibuja canal continuo con contorno de pluma."""
    StyleConfig = {
        "Cmap": Cmap,
        "RobustPercentiles": [2, 98],
        "CenterZero": CenterZero,
        "ColorbarLabel": ColorbarLabel,
    }

    ImageHandle = Axis.imshow(
        Image,
        cmap=Cmap,
        norm=BuildNormalize(Image, StyleConfig),
        interpolation=VisualConfig["Visualization"]["Image"].get("Interpolation", "nearest"),
    )

    DrawPlumeContour(Axis, Plume, VisualConfig, PlumeRole="GroundTruth")
    Axis.set_title(Title)
    FormatImageAxis(Axis, VisualConfig)
    AddColorbar(Figure, Axis, ImageHandle, ColorbarLabel, VisualConfig)


def PlotFeaturePreview(
    Target: np.ndarray,
    Reference: np.ndarray,
    CH4: np.ndarray | None,
    Plume: np.ndarray,
    FeatureDictionary: dict[str, np.ndarray],
    FeatureConfig: str,
    SampleId: str,
    VisualConfig: dict[str, Any],
    SavePath: str | Path | None = None,
    ShowFigure: bool = False,
):
    """Genera figura de preview de features."""
    ImageMap: dict[str, np.ndarray | None] = {
        "TargetRGB": BuildRgbFromSentinel(Target, VisualConfig),
        "ReferenceRGB": BuildRgbFromSentinel(Reference, VisualConfig),
        "TargetSWIR": BuildSwirComposite(Target, VisualConfig),
        "ReferenceSWIR": BuildSwirComposite(Reference, VisualConfig),
        "CH4": CH4,
        "Plume": Plume,
    }

    ImageMap.update(FeatureDictionary)

    Panels = BuildFeaturePanelSpecs(FeatureConfig=FeatureConfig, IncludeCH4=CH4 is not None)

    Columns = 4
    Rows = int(np.ceil(len(Panels) / Columns))

    Figure, Axes = plt.subplots(Rows, Columns, figsize=(4.3 * Columns, 4.2 * Rows))
    Axes = np.asarray(Axes).ravel()

    UsedRoles = ["GroundTruth"]

    for Index, Panel in enumerate(Panels):
        Axis = Axes[Index]
        Image = ImageMap.get(Panel["ImageKey"])

        if Image is None:
            Axis.axis("off")
            Axis.set_title(f"{Panel['Title']} unavailable")
            continue

        if Panel["Kind"] == "rgb":
            Axis.imshow(Image)
            DrawPlumeContour(Axis, Plume, VisualConfig, PlumeRole="GroundTruth")
            Axis.set_title(Panel["Title"])
            FormatImageAxis(Axis, VisualConfig)

        elif Panel["Kind"] == "channel":
            PlotChannel(
                Figure=Figure,
                Axis=Axis,
                Image=Image,
                Plume=Plume,
                Title=Panel["Title"],
                Cmap=Panel.get("Cmap", "viridis"),
                CenterZero=bool(Panel.get("CenterZero", False)),
                ColorbarLabel=Panel.get("ColorbarLabel", "Value"),
                VisualConfig=VisualConfig,
            )

        elif Panel["Kind"] == "mask":
            Axis.imshow(Image > 0, cmap=Panel.get("Cmap", "gray"), vmin=0, vmax=1)
            DrawPlumeContour(Axis, Image, VisualConfig, PlumeRole="GroundTruth")
            Axis.set_title(Panel["Title"])
            FormatImageAxis(Axis, VisualConfig)

        else:
            raise ValueError(f"Kind no reconocido: {Panel['Kind']}")

    for Axis in Axes[len(Panels):]:
        Axis.axis("off")

    Figure.suptitle(f"{FeatureConfig} feature preview - SampleId {SampleId}")
    AddPlumeLegend(Figure, VisualConfig, UsedRoles, Location="lower center")
    Figure.tight_layout(rect=[0, 0.05, 1, 0.95])

    if SavePath is not None:
        SavePath = Path(SavePath)
        SavePath.parent.mkdir(parents=True, exist_ok=True)
        Figure.savefig(
            SavePath,
            dpi=VisualConfig["Visualization"].get("Dpi", 300),
            bbox_inches="tight",
            facecolor=VisualConfig["Visualization"].get("FigureFaceColor", "white"),
        )

    if ShowFigure:
        plt.show()
    else:
        plt.close(Figure)

    return Figure
