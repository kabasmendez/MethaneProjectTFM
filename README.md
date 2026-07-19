# MethaneProjectTFM

Proyecto Python limpio y reproducible para el Trabajo Final de Máster sobre segmentación automática de plumas de metano en imágenes Sentinel-2.

## Principios

- No se copia código del proyecto anterior.
- El proyecto anterior se usa solo como referencia técnica.
- Todas las salidas se organizan por experimento.
- Todo experimento usa un único `RunTag` con formato `ExpDDHHMM`.
- Las figuras, mapas, paneles y colorbars se controlan desde `Configs/VisualizationConfig.yaml`.

## Estructura de salidas

```text
Outputs/Experiments/ExpDDHHMM/
├── Tables/
├── ConfigA/
├── ConfigB/
├── ConfigC/
├── Compare/
├── Reports/
├── Logs/
└── Audit/


Ahora crea los módulos visuales:

```bash
cat > Source/VisualizationStyle.py <<'EOF'
"""
VisualizationStyle.py

Carga y aplica el estándar visual global de MethaneProjectTFM.

Todas las figuras deben depender de Configs/VisualizationConfig.yaml.
Ningún script debe definir fuentes, DPI, tamaños o estilos a mano.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def ApplyMatplotlibStyle(VisualConfig: dict[str, Any]) -> None:
    """Aplica estilo global de matplotlib desde VisualizationConfig."""
    Visualization = VisualConfig["Visualization"]
    FontSizes = Visualization["FontSizes"]

    plt.rcParams.update(
        {
            "font.family": Visualization.get("FontFamily", "DejaVu Sans"),
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
