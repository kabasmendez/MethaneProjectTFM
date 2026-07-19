"""
VisualizeSamples.py

Visualización estandarizada de muestras Sentinel-2/TACO.

Reglas visuales:
- La pluma ground truth se dibuja siempre como contorno, sin relleno.
- Los colores de contorno se leen desde VisualizationConfig.yaml.
- Cada matriz de figuras debe incluir leyenda de contornos.
- CH4 se usa solo como apoyo visual, no como input del modelo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from Source.ColorMaps import BuildNormalize, BuildRgbImage
from Source.MapPlotUtils import (
    AddColorbar,
    AddMetadataText,
    AddPlumeLegend,
    DrawPlumeContour,
    FormatImageAxis,
)


S2_BAND_INDEX = {
    "B01": 0,
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}


def GetBand(Image: np.ndarray, BandName: str) -> np.ndarray:
    """Extrae banda Sentinel-2 por nombre."""
    if BandName not in S2_BAND_INDEX:
        raise ValueError(f"Banda Sentinel-2 no reconocida: {BandName}")

    BandIndex = S2_BAND_INDEX[BandName]

    if Image.ndim != 3:
        raise ValueError(f"Imagen debe tener forma (Bands, H, W). Forma recibida: {Image.shape}")

    if Image.shape[0] <= BandIndex:
        raise ValueError(f"Imagen tiene {Image.shape[0]} bandas. No se puede extraer {BandName}")

    return Image[BandIndex].astype(np.float32)


def BuildRgbFromSentinel(Image: np.ndarray, VisualConfig: dict[str, Any]) -> np.ndarray:
    """Construye RGB natural B04/B03/B02."""
    RGBConfig = VisualConfig["ColorMaps"]["RGB"]
    Lower, Upper = RGBConfig.get("RobustPercentiles", [2, 98])
    Gamma = float(RGBConfig.get("Gamma", 1.0))

    return BuildRgbImage(
        GetBand(Image, "B04"),
        GetBand(Image, "B03"),
        GetBand(Image, "B02"),
        LowerPercentile=float(Lower),
        UpperPercentile=float(Upper),
        Gamma=Gamma,
    )


def BuildSwirComposite(Image: np.ndarray, VisualConfig: dict[str, Any]) -> np.ndarray:
    """Construye falso color SWIR B12/B11/B8A."""
    SWIRConfig = VisualConfig["ColorMaps"]["SwirComposite"]
    Lower, Upper = SWIRConfig.get("RobustPercentiles", [2, 98])
    Gamma = float(SWIRConfig.get("Gamma", 1.0))

    return BuildRgbImage(
        GetBand(Image, "B12"),
        GetBand(Image, "B11"),
        GetBand(Image, "B8A"),
        LowerPercentile=float(Lower),
        UpperPercentile=float(Upper),
        Gamma=Gamma,
    )


def BuildMetadataLines(SampleData: dict[str, Any]) -> list[str]:
    """Construye texto corto de metadatos."""
    Metadata = SampleData.get("Metadata", {})
    SampleId = SampleData.get("SampleId", "")

    Lines = [f"SampleId: {SampleId}"]

    WindU = Metadata.get("meteo:wind_u")
    WindV = Metadata.get("meteo:wind_v")
    SolarZenith = Metadata.get("satellite:sza")
    Observability = Metadata.get("quality:observability")
    Clear = Metadata.get("quality:percentage_clear")

    try:
        if WindU is not None and WindV is not None:
            WindSpeed = float(np.sqrt(float(WindU) ** 2 + float(WindV) ** 2))
            Lines.append(
                f"Wind: U={float(WindU):.2f}, V={float(WindV):.2f}, "
                f"speed={WindSpeed:.2f} m/s"
            )
    except Exception:
        Lines.append(f"Wind: U={WindU}, V={WindV}")

    try:
        if SolarZenith is not None:
            Lines.append(f"SZA: {float(SolarZenith):.2f} deg")
    except Exception:
        Lines.append(f"SZA: {SolarZenith}")

    if Observability is not None:
        Lines.append(f"Obs: {Observability}")

    if Clear is not None:
        try:
            Lines.append(f"Clear: {float(Clear):.1f}%")
        except Exception:
            Lines.append(f"Clear: {Clear}")

    return Lines


def GetSampleImageMap(SampleData: dict[str, Any], VisualConfig: dict[str, Any]) -> dict[str, np.ndarray | None]:
    """Construye diccionario de imágenes estándar para paneles."""
    Target = SampleData["Target"]
    Reference = SampleData["Reference"]

    return {
        "TargetRGB": BuildRgbFromSentinel(Target, VisualConfig),
        "ReferenceRGB": BuildRgbFromSentinel(Reference, VisualConfig),
        "TargetSWIR": BuildSwirComposite(Target, VisualConfig),
        "ReferenceSWIR": BuildSwirComposite(Reference, VisualConfig),
        "CH4": SampleData.get("CH4"),
        "Plume": SampleData.get("Plume"),
    }


def PlotRgbPanel(
    Axis,
    Image: np.ndarray,
    Plume: np.ndarray,
    Title: str,
    VisualConfig: dict[str, Any],
    PlumeRole: str = "GroundTruth",
) -> None:
    """Panel RGB con contorno de pluma sin relleno."""
    Axis.imshow(Image)
    DrawPlumeContour(Axis, Plume, VisualConfig, PlumeRole=PlumeRole)
    Axis.set_title(Title)
    FormatImageAxis(Axis, VisualConfig)


def PlotChannelPanel(
    Figure,
    Axis,
    Image: np.ndarray,
    Plume: np.ndarray,
    Title: str,
    Cmap: str,
    VisualConfig: dict[str, Any],
    PlumeRole: str = "GroundTruth",
    Normalize: bool = True,
    ColorbarLabel: str = "",
) -> None:
    """Panel de canal continuo con contorno de pluma sin relleno."""
    if Image is None:
        Axis.axis("off")
        Axis.set_title(f"{Title} unavailable")
        return

    if Normalize:
        StyleConfig = {
            "Cmap": Cmap,
            "RobustPercentiles": [2, 98],
            "CenterZero": Cmap in ["coolwarm", "RdBu_r", "RdYlBu_r"],
            "Colorbar": True,
            "ColorbarLabel": ColorbarLabel or "Value",
        }
        Norm = BuildNormalize(Image, StyleConfig)
        ImageHandle = Axis.imshow(
            Image,
            cmap=Cmap,
            norm=Norm,
            interpolation=VisualConfig["Visualization"]["Image"].get("Interpolation", "nearest"),
        )
    else:
        ImageHandle = Axis.imshow(
            Image,
            cmap=Cmap,
            interpolation=VisualConfig["Visualization"]["Image"].get("Interpolation", "nearest"),
        )

    DrawPlumeContour(Axis, Plume, VisualConfig, PlumeRole=PlumeRole)
    Axis.set_title(Title)
    FormatImageAxis(Axis, VisualConfig)

    AddColorbar(Figure, Axis, ImageHandle, ColorbarLabel or "Value", VisualConfig)


def PlotMaskPanel(
    Axis,
    Mask: np.ndarray,
    Title: str,
    VisualConfig: dict[str, Any],
    PlumeRole: str = "GroundTruth",
) -> None:
    """
    Panel de máscara.

    La máscara puede mostrarse como fondo gris para inspección,
    pero la pluma oficial se marca con contorno sin relleno.
    """
    Axis.imshow(Mask > 0, cmap="gray", vmin=0, vmax=1)
    DrawPlumeContour(Axis, Mask, VisualConfig, PlumeRole=PlumeRole)
    Axis.set_title(Title)
    FormatImageAxis(Axis, VisualConfig)


def PlotPanel(
    Figure,
    Axis,
    PanelSpec: dict[str, Any],
    ImageMap: dict[str, np.ndarray | None],
    Plume: np.ndarray,
    VisualConfig: dict[str, Any],
) -> str | None:
    """Dibuja un panel declarativo y devuelve el PlumeRole usado."""
    Kind = PanelSpec["Kind"]
    ImageKey = PanelSpec["Image"]
    Title = PanelSpec.get("Title", ImageKey)
    PlumeRole = PanelSpec.get("PlumeRole", "GroundTruth")

    if Kind == "rgb":
        PlotRgbPanel(
            Axis=Axis,
            Image=ImageMap[ImageKey],
            Plume=Plume,
            Title=Title,
            VisualConfig=VisualConfig,
            PlumeRole=PlumeRole,
        )
        return PlumeRole

    if Kind == "channel":
        PlotChannelPanel(
            Figure=Figure,
            Axis=Axis,
            Image=ImageMap[ImageKey],
            Plume=Plume,
            Title=Title,
            Cmap=PanelSpec.get("Cmap", "viridis"),
            VisualConfig=VisualConfig,
            PlumeRole=PlumeRole,
            Normalize=bool(PanelSpec.get("Normalize", True)),
            ColorbarLabel=PanelSpec.get("ColorbarLabel", ""),
        )
        return PlumeRole

    if Kind == "mask":
        PlotMaskPanel(
            Axis=Axis,
            Mask=ImageMap[ImageKey],
            Title=Title,
            VisualConfig=VisualConfig,
            PlumeRole=PlumeRole,
        )
        return PlumeRole

    raise ValueError(f"Kind de panel no reconocido: {Kind}")


def PlotSampleOverview(
    SampleData: dict[str, Any],
    VisualConfig: dict[str, Any],
    SavePath=None,
    ShowFigure: bool = False,
):
    """
    Crea figura estándar de inspección de muestra.

    Paneles definidos en VisualizationConfig.yaml:
    PanelTemplates.SampleOverviewBase
    """
    Plume = SampleData["Plume"]
    ImageMap = GetSampleImageMap(SampleData, VisualConfig)
    Panels = VisualConfig["PanelTemplates"]["SampleOverviewBase"]

    Columns = 3
    Rows = int(np.ceil(len(Panels) / Columns)) + 1

    Figure, Axes = plt.subplots(Rows, Columns, figsize=(15, 5 * Rows))
    Axes = Axes.ravel()

    UsedPlumeRoles = []

    for PanelIndex, PanelSpec in enumerate(Panels):
        Role = PlotPanel(
            Figure=Figure,
            Axis=Axes[PanelIndex],
            PanelSpec=PanelSpec,
            ImageMap=ImageMap,
            Plume=Plume,
            VisualConfig=VisualConfig,
        )
        if Role is not None:
            UsedPlumeRoles.append(Role)

    MetadataAxis = Axes[len(Panels)]
    MetadataAxis.axis("off")
    AddMetadataText(MetadataAxis, BuildMetadataLines(SampleData), VisualConfig)

    for Axis in Axes[len(Panels) + 1:]:
        Axis.axis("off")

    Figure.suptitle(
        "Sample overview",
        fontsize=VisualConfig["Visualization"]["FontSizes"].get("Title", 12),
    )

    AddPlumeLegend(Figure, VisualConfig, UsedPlumeRoles, Location="lower center")

    Figure.tight_layout(rect=[0, 0.04, 1, 0.96])

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


def PlotSampleGrid(
    SampleDataList: list[dict[str, Any]],
    VisualConfig: dict[str, Any],
    SavePath=None,
    ShowFigure: bool = False,
):
    """Crea grid compacto con Target RGB + contorno GT sin relleno."""
    if len(SampleDataList) == 0:
        raise ValueError("SampleDataList está vacío.")

    Columns = min(4, len(SampleDataList))
    Rows = int(np.ceil(len(SampleDataList) / Columns))

    Figure, Axes = plt.subplots(Rows, Columns, figsize=(4 * Columns, 4 * Rows))

    if not isinstance(Axes, np.ndarray):
        Axes = np.array([Axes])

    Axes = Axes.ravel()

    UsedPlumeRoles = ["GroundTruth"]

    for AxisIndex, Axis in enumerate(Axes):
        if AxisIndex >= len(SampleDataList):
            Axis.axis("off")
            continue

        SampleData = SampleDataList[AxisIndex]
        TargetRgb = BuildRgbFromSentinel(SampleData["Target"], VisualConfig)
        Plume = SampleData["Plume"]

        Axis.imshow(TargetRgb)
        DrawPlumeContour(Axis, Plume, VisualConfig, PlumeRole="GroundTruth")
        Axis.set_title(str(SampleData["SampleId"])[:18])
        FormatImageAxis(Axis, VisualConfig)

    Figure.suptitle("Sample grid: Target RGB with ground-truth contour")
    AddPlumeLegend(Figure, VisualConfig, UsedPlumeRoles, Location="lower center")
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
