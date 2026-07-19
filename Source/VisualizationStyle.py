"""
VisualizationStyle.py

Carga y aplica el estándar visual global de MethaneProjectTFM.

Todas las figuras deben depender de Configs/VisualizationConfig.yaml.
Ningún script debe definir fuentes, DPI, tamaños o estilos a mano.

Fuente oficial del proyecto:
- Montserrat, si está disponible en el entorno.
- DejaVu Sans como fallback para evitar fallos de ejecución.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import yaml


def LoadVisualizationConfig(ConfigPath: str | Path = "Configs/VisualizationConfig.yaml") -> dict[str, Any]:
    """Carga el archivo YAML de configuración visual."""
    ConfigPath = Path(ConfigPath)

    if not ConfigPath.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración visual: {ConfigPath}")

    with ConfigPath.open("r", encoding="utf-8") as File:
        Config = yaml.safe_load(File)

    if not isinstance(Config, dict):
        raise ValueError(f"El archivo visual no contiene un diccionario válido: {ConfigPath}")

    RequiredSections = ["Visualization", "ColorMaps", "FeatureStyles", "Panels"]
    MissingSections = [Section for Section in RequiredSections if Section not in Config]

    if MissingSections:
        raise KeyError(f"Faltan secciones obligatorias en VisualizationConfig: {MissingSections}")

    return Config


def IsFontAvailable(FontFamily: str) -> bool:
    """Comprueba si una fuente está disponible para Matplotlib."""
    AvailableFonts = {Font.name for Font in font_manager.fontManager.ttflist}
    return FontFamily in AvailableFonts


def ResolveFontFamily(VisualConfig: dict[str, Any]) -> str:
    """
    Devuelve la fuente que se usará realmente.

    Si Montserrat no está instalada, usa FallbackFontFamily y emite warning.
    """
    Visualization = VisualConfig["Visualization"]
    RequestedFont = Visualization.get("FontFamily", "Montserrat")
    FallbackFont = Visualization.get("FallbackFontFamily", "DejaVu Sans")

    if IsFontAvailable(RequestedFont):
        return RequestedFont

    warnings.warn(
        f"La fuente '{RequestedFont}' no está disponible en este entorno. "
        f"Se usará fallback '{FallbackFont}'. Para usar Montserrat, instálala en el sistema "
        f"o en el entorno gráfico antes de generar figuras finales.",
        UserWarning,
    )

    return FallbackFont


def ApplyMatplotlibStyle(VisualConfig: dict[str, Any]) -> None:
    """Aplica estilo global de matplotlib desde VisualizationConfig."""
    Visualization = VisualConfig["Visualization"]
    FontSizes = Visualization["FontSizes"]
    FontFamily = ResolveFontFamily(VisualConfig)
    TextColors = Visualization.get("TextColors", {})

    plt.rcParams.update(
        {
            "font.family": FontFamily,
            "figure.facecolor": Visualization.get("FigureFaceColor", "white"),
            "axes.facecolor": Visualization.get("AxesFaceColor", "white"),
            "axes.titlesize": FontSizes.get("Title", 12),
            "axes.labelsize": FontSizes.get("AxisLabel", 9),
            "xtick.labelsize": FontSizes.get("TickLabel", 8),
            "ytick.labelsize": FontSizes.get("TickLabel", 8),
            "legend.fontsize": FontSizes.get("Annotation", 8),
            "savefig.dpi": Visualization.get("Dpi", 300),
            "savefig.bbox": "tight",
            "savefig.facecolor": Visualization.get("FigureFaceColor", "white"),
            "axes.titlecolor": TextColors.get("Title", "black"),
            "axes.labelcolor": TextColors.get("AxisLabel", "black"),
            "xtick.color": TextColors.get("TickLabel", "black"),
            "ytick.color": TextColors.get("TickLabel", "black"),
            "text.color": TextColors.get("Annotation", "black"),
            "legend.labelcolor": TextColors.get("LegendText", "black"),
        }
    )


def GetFigureSize(VisualConfig: dict[str, Any], SizeName: str) -> tuple[float, float]:
    """Devuelve tamaño de figura por nombre estándar."""
    FigureSizes = VisualConfig["Visualization"]["FigureSizes"]

    if SizeName not in FigureSizes:
        raise KeyError(f"Tamaño de figura no definido en VisualizationConfig: {SizeName}")

    Width, Height = FigureSizes[SizeName]
    return float(Width), float(Height)


def GetSaveParameters(VisualConfig: dict[str, Any]) -> dict[str, Any]:
    """Devuelve parámetros estándar de guardado."""
    Visualization = VisualConfig["Visualization"]

    return {
        "dpi": int(Visualization.get("Dpi", 300)),
        "bbox_inches": "tight",
        "facecolor": Visualization.get("FigureFaceColor", "white"),
    }
