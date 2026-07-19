#!/usr/bin/env python3
"""
Step16BuildComparisonHtmlReport.py

Dashboard HTML comparativo para 2 o máximo 3 ejecuciones.

Diseño:
- Comparación visual y compacta.
- Plotly interactivo.
- Montserrat.
- Sin tablas grandes visibles.
- Tablas completas dentro de <details>.
- Comparación pensada para pocos modelos.
- Muestra algunos casos visuales fijos si existen figuras de Step13.
"""

from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import BuildAuditRecord, WriteJson


FONT = "Montserrat, Arial, sans-serif"

PALETTE = {
    "navy": "#0B2A63",
    "blue": "#1F65C8",
    "blue2": "#2C7DE1",
    "cyan": "#5BB7E8",
    "cyan2": "#8BD9E8",
    "pale": "#EDF5FF",
    "grid": "#E5EEF8",
    "text": "#16325C",
    "muted": "#4F6787",
    "white": "#FFFFFF",
}

MODEL_COLORS = [
    "#5BB7E8",
    "#0B2A63",
    "#2C7DE1",
]


def Escape(Value: Any) -> str:
    if Value is None:
        return ""
    try:
        if pd.isna(Value):
            return ""
    except Exception:
        pass
    return html.escape(str(Value))


def FormatNumber(Value: Any, Digits: int = 4) -> str:
    try:
        if pd.isna(Value):
            return "N/D"
    except Exception:
        pass

    try:
        return f"{float(Value):.{Digits}f}"
    except Exception:
        return Escape(Value)


def SafeReadCsv(PathItem: Path, Required: bool = False) -> pd.DataFrame:
    if not PathItem.exists():
        if Required:
            raise FileNotFoundError(PathItem)
        return pd.DataFrame()
    return pd.read_csv(PathItem)


def EnsureDirectory(PathItem: Path) -> Path:
    PathItem.mkdir(parents=True, exist_ok=True)
    return PathItem


def CopyPlotlyJs(ReportsDirectory: Path) -> str:
    import plotly

    Source = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    if not Source.exists():
        raise FileNotFoundError(f"No encontré plotly.min.js en {Source}")

    TargetDirectory = ReportsDirectory / "assets" / "plotly"
    EnsureDirectory(TargetDirectory)

    Target = TargetDirectory / "plotly.min.js"
    shutil.copy2(Source, Target)

    return "assets/plotly/plotly.min.js"


def PlotToHtml(Figure: go.Figure) -> str:
    return Figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "MethaneProjectTFM_comparison_plot",
                "height": 900,
                "width": 1400,
                "scale": 2,
            },
        },
    )


def ApplyLayout(Figure: go.Figure, Title: str, Height: int = 460) -> go.Figure:
    Figure.update_layout(
        title=Title,
        height=Height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, color=PALETTE["text"], size=15),
        title_font=dict(family=FONT, color=PALETTE["navy"], size=24),
        margin=dict(l=70, r=30, t=76, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT, size=13),
        ),
    )

    Figure.update_xaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        linecolor=PALETTE["cyan"],
        linewidth=1.5,
        tickfont=dict(family=FONT, size=14, color=PALETTE["text"]),
        title_font=dict(family=FONT, size=15, color=PALETTE["text"]),
    )

    Figure.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        linecolor=PALETTE["cyan"],
        linewidth=1.5,
        tickfont=dict(family=FONT, size=14, color=PALETTE["text"]),
        title_font=dict(family=FONT, size=15, color=PALETTE["text"]),
    )

    return Figure


def MakeShortModelName(Row: pd.Series) -> str:
    if "ModelRunId" in Row and pd.notna(Row["ModelRunId"]):
        return str(Row["ModelRunId"]).replace("TransformerUNet_", "T-UNet_").replace("EnhancedUNet_", "E-UNet_").replace("SimpleUNet_", "S-UNet_")
    if "ItemId" in Row:
        return str(Row["ItemId"])
    return "Model"


def BuildKpiCards(Summary: pd.DataFrame) -> str:
    if Summary.empty:
        return "<p class='muted'>No hay datos de comparación.</p>"

    Cards = []

    for Index, (_, Row) in enumerate(Summary.iterrows()):
        Color = MODEL_COLORS[Index % len(MODEL_COLORS)]
        ModelName = MakeShortModelName(Row)

        Cards.append(
            f"""
            <article class="model-card" style="--model-color:{Color};">
                <div class="model-chip">Modelo {Index + 1}</div>
                <h3>{Escape(ModelName)}</h3>
                <div class="model-metrics">
                    <div><span>Mean Dice</span><strong>{FormatNumber(Row.get("MeanDice", ""), 4)}</strong></div>
                    <div><span>Mean IoU</span><strong>{FormatNumber(Row.get("MeanIoU", ""), 4)}</strong></div>
                    <div><span>Global Dice</span><strong>{FormatNumber(Row.get("GlobalDice", ""), 4)}</strong></div>
                    <div><span>Global IoU</span><strong>{FormatNumber(Row.get("GlobalIoU", ""), 4)}</strong></div>
                </div>
            </article>
            """
        )

    return "<div class='model-grid'>" + "\n".join(Cards) + "</div>"


def BuildWinnerSummary(Summary: pd.DataFrame) -> str:
    if Summary.empty:
        return ""

    Metrics = ["MeanDice", "MeanIoU", "GlobalDice", "GlobalIoU", "MeanPrecision", "MeanRecall"]
    Available = [Metric for Metric in Metrics if Metric in Summary.columns]

    Items = []

    for Metric in Available:
        Best = Summary.sort_values(Metric, ascending=False).iloc[0]
        Items.append(
            f"""
            <div class="winner-item">
                <span>{Escape(Metric)}</span>
                <strong>{Escape(MakeShortModelName(Best))}</strong>
                <em>{FormatNumber(Best.get(Metric), 4)}</em>
            </div>
            """
        )

    return "<div class='winner-grid'>" + "\n".join(Items) + "</div>"


def BuildMetricBarChart(Summary: pd.DataFrame) -> str:
    if Summary.empty:
        return "<p class='muted'>No disponible.</p>"

    Metrics = ["MeanDice", "MeanIoU", "GlobalDice", "GlobalIoU", "MeanPrecision", "MeanRecall"]
    Labels = ["Mean<br>Dice", "Mean<br>IoU", "Global<br>Dice", "Global<br>IoU", "Precision", "Recall"]
    AvailablePairs = [(m, l) for m, l in zip(Metrics, Labels) if m in Summary.columns]

    Figure = go.Figure()

    for Index, (_, Row) in enumerate(Summary.iterrows()):
        Color = MODEL_COLORS[Index % len(MODEL_COLORS)]
        Values = [float(Row.get(Metric, np.nan)) for Metric, _ in AvailablePairs]

        Figure.add_trace(
            go.Bar(
                x=[Label for _, Label in AvailablePairs],
                y=Values,
                name=MakeShortModelName(Row),
                marker=dict(color=Color, line=dict(color=PALETTE["navy"], width=1.0)),
                text=[FormatNumber(v, 3) for v in Values],
                textposition="outside",
                textfont=dict(family=FONT, size=13, color=PALETTE["text"]),
            )
        )

    Figure.update_yaxes(title="Valor de métrica", range=[0, 1])
    Figure.update_xaxes(title="")

    ApplyLayout(Figure, "Comparación de métricas principales", Height=500)
    return PlotToHtml(Figure)


def BuildMetricHeatmap(Summary: pd.DataFrame) -> str:
    if Summary.empty:
        return "<p class='muted'>No disponible.</p>"

    Metrics = ["MeanDice", "MeanIoU", "GlobalDice", "GlobalIoU", "MeanPrecision", "MeanRecall"]
    Available = [Metric for Metric in Metrics if Metric in Summary.columns]

    if not Available:
        return "<p class='muted'>No disponible.</p>"

    Models = [MakeShortModelName(Row) for _, Row in Summary.iterrows()]
    Values = Summary[Available].astype(float).values

    Figure = go.Figure(
        data=go.Heatmap(
            z=Values,
            x=[m.replace("Mean", "Mean ").replace("Global", "Global ") for m in Available],
            y=Models,
            colorscale=[
                [0.0, "#F1F6FB"],
                [0.5, "#5BB7E8"],
                [1.0, "#0B2A63"],
            ],
            zmin=0,
            zmax=max(0.01, float(np.nanmax(Values))),
            text=np.round(Values, 4),
            texttemplate="%{text:.4f}",
            textfont=dict(family=FONT, size=13, color=PALETTE["text"]),
            colorbar=dict(title="Valor"),
        )
    )

    ApplyLayout(Figure, "Mapa compacto de desempeño", Height=340 + 70 * len(Models))
    return PlotToHtml(Figure)


def BuildTrainingCurves(ProjectRoot: Path, Summary: pd.DataFrame) -> str:
    if Summary.empty:
        return "<p class='muted'>No disponible.</p>"

    Figure = go.Figure()
    Added = False

    for Index, (_, Row) in enumerate(Summary.iterrows()):
        RunTag = str(Row.get("RunTag", ""))
        FeatureConfig = str(Row.get("FeatureConfig", ""))
        ModelRunId = str(Row.get("ModelRunId", ""))

        HistoryPath = (
            ProjectRoot
            / "Outputs"
            / "Experiments"
            / RunTag
            / FeatureConfig
            / ModelRunId
            / "Metrics"
            / "TrainingHistory.csv"
        )

        if not HistoryPath.exists():
            continue

        History = pd.read_csv(HistoryPath)

        if History.empty or "Epoch" not in History.columns:
            continue

        Color = MODEL_COLORS[Index % len(MODEL_COLORS)]
        Name = MakeShortModelName(Row)

        if "ValidationMeanDice" in History.columns:
            Figure.add_trace(
                go.Scatter(
                    x=History["Epoch"],
                    y=History["ValidationMeanDice"],
                    mode="lines+markers",
                    name=f"{Name} · Val Dice",
                    line=dict(color=Color, width=4),
                    marker=dict(size=7),
                )
            )
            Added = True

        if "TrainMeanDice" in History.columns:
            Figure.add_trace(
                go.Scatter(
                    x=History["Epoch"],
                    y=History["TrainMeanDice"],
                    mode="lines",
                    name=f"{Name} · Train Dice",
                    line=dict(color=Color, width=3, dash="dash"),
                    opacity=0.75,
                )
            )
            Added = True

    if not Added:
        return "<p class='muted'>No hay TrainingHistory disponible.</p>"

    Figure.update_xaxes(title="Época")
    Figure.update_yaxes(title="Dice", range=[0, 1])

    ApplyLayout(Figure, "Curvas Train/Validation Dice", Height=500)
    return PlotToHtml(Figure)


def BuildFixedCaseVisuals(ProjectRoot: Path, Summary: pd.DataFrame, MaxCases: int = 3) -> str:
    """
    Muestra comparación visual sobre los mismos SampleId, usando figuras ya generadas por Step13.
    Solo carga pocas imágenes para no hacer pesado el HTML.
    """
    if Summary.empty:
        return "<p class='muted'>No hay modelos.</p>"

    FigureTables = []

    for _, Row in Summary.iterrows():
        ModelRoot = (
            ProjectRoot
            / "Outputs"
            / "Experiments"
            / str(Row["RunTag"])
            / str(Row["FeatureConfig"])
            / str(Row["ModelRunId"])
        )

        IndexPath = ModelRoot / "Tables" / "PredictionFigureIndex.csv"

        if not IndexPath.exists():
            continue

        Index = pd.read_csv(IndexPath)
        Index["ModelRunId"] = Row["ModelRunId"]
        Index["ShortModelName"] = MakeShortModelName(Row)
        Index["ModelRoot"] = str(ModelRoot)
        FigureTables.append(Index)

    if not FigureTables:
        return "<p class='muted'>No hay figuras de Step13 disponibles para los modelos comparados.</p>"

    Figures = pd.concat(FigureTables, ignore_index=True)

    if "CaseGroup" in Figures.columns:
        Fixed = Figures[Figures["CaseGroup"] == "FixedComparisonCases"].copy()
    else:
        Fixed = Figures.copy()

    if Fixed.empty:
        return "<p class='muted'>No hay FixedComparisonCases disponibles.</p>"

    # Solo casos presentes en al menos dos modelos.
    Counts = Fixed.groupby("SampleId")["ModelRunId"].nunique().sort_values(ascending=False)
    CandidateIds = Counts[Counts >= min(2, len(Summary))].index.tolist()

    if not CandidateIds:
        CandidateIds = Fixed["SampleId"].drop_duplicates().head(MaxCases).tolist()

    SampleIds = CandidateIds[:MaxCases]

    Blocks = []

    for SampleId in SampleIds:
        Subset = Fixed[Fixed["SampleId"] == SampleId].copy()

        Cards = []

        for _, Row in Subset.iterrows():
            FigurePath = Path(str(Row["ModelRoot"])) / str(Row["FigurePath"])

            if not FigurePath.exists():
                continue

            # Desde Outputs/Comparisons/<Tag>/Reports/ComparisonReport.html
            # hasta raíz del proyecto: ../../../
            ProjectRelative = FigurePath.relative_to(ProjectRoot)
            Src = "../../../" + str(ProjectRelative).replace("\\", "/")

            Caption = (
                f"{Row.get('ShortModelName', '')} · "
                f"Dice={FormatNumber(Row.get('Dice', ''), 3)} · "
                f"IoU={FormatNumber(Row.get('IoU', ''), 3)}"
            )

            Cards.append(
                f"""
                <article class="image-card">
                    <a href="{Escape(Src)}" target="_blank">
                        <img loading="lazy" src="{Escape(Src)}" alt="{Escape(Caption)}"/>
                    </a>
                    <figcaption>{Escape(Caption)}</figcaption>
                </article>
                """
            )

        if Cards:
            Blocks.append(
                f"""
                <div class="case-block">
                    <h3>SampleId: {Escape(SampleId)}</h3>
                    <div class="image-grid">
                        {''.join(Cards)}
                    </div>
                </div>
                """
            )

    if not Blocks:
        return "<p class='muted'>No fue posible cargar imágenes comparativas.</p>"

    return "\n".join(Blocks)


def TableToHtml(Table: pd.DataFrame, MaxRows: int = 30) -> str:
    if Table.empty:
        return "<p class='muted'>No disponible.</p>"

    Work = Table.head(MaxRows).copy()

    Header = "".join(f"<th>{Escape(Column)}</th>" for Column in Work.columns)

    Rows = []
    for _, Row in Work.iterrows():
        Cells = ""
        for Column in Work.columns:
            Value = Row[Column]
            Text = FormatNumber(Value) if isinstance(Value, float) else Escape(Value)
            Cells += f"<td>{Text}</td>"
        Rows.append(f"<tr>{Cells}</tr>")

    return f"""
    <div class="table-wrap">
        <table class="compact-table">
            <thead><tr>{Header}</tr></thead>
            <tbody>{''.join(Rows)}</tbody>
        </table>
    </div>
    """


def BuildDetailsTables(Summary: pd.DataFrame, BySample: pd.DataFrame, LongBySample: pd.DataFrame) -> str:
    return f"""
    <details class="details-block">
        <summary>Tabla resumen completa</summary>
        {TableToHtml(Summary, MaxRows=20)}
    </details>

    <details class="details-block">
        <summary>Métricas por muestra, formato ancho</summary>
        {TableToHtml(BySample, MaxRows=40)}
    </details>

    <details class="details-block">
        <summary>Métricas por muestra, formato largo</summary>
        {TableToHtml(LongBySample, MaxRows=80)}
    </details>
    """


def BuildHtml(
    ProjectRoot: Path,
    ComparisonTag: str,
    PlotlySrc: str,
    Summary: pd.DataFrame,
    BySample: pd.DataFrame,
    LongBySample: pd.DataFrame,
    MaxVisualCases: int,
) -> str:
    Runs = len(Summary)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>MethaneProjectTFM · Comparison · {Escape(ComparisonTag)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="{Escape(PlotlySrc)}"></script>
<style>
{BuildCss()}
</style>
</head>

<body>
<div class="container">

<section class="hero">
    <div>
        <p class="eyebrow">MethaneProjectTFM · model comparison</p>
        <h1>{Escape(ComparisonTag)}</h1>
        <p class="hero-text">
            Dashboard compacto para comparar dos o tres ejecuciones.
            El foco está en métricas principales, curvas de entrenamiento y casos visuales fijos.
        </p>
    </div>
    <div class="hero-meta">
        <div><span>Runs</span>{Runs}</div>
        <div><span>Scope</span>2–3 modelos</div>
        <div><span>Metric</span>Dice / IoU</div>
        <div><span>Visual cases</span>{MaxVisualCases}</div>
    </div>
</section>

<section class="section">
    <h2>Resumen por modelo</h2>
    {BuildKpiCards(Summary)}
</section>

<section class="section">
    <h2>Ganadores por métrica</h2>
    {BuildWinnerSummary(Summary)}
</section>

<section class="section">
    <h2>Comparación principal</h2>
    <div class="two-col">
        <div class="plot-card">{BuildMetricBarChart(Summary)}</div>
        <div class="plot-card">{BuildMetricHeatmap(Summary)}</div>
    </div>
</section>

<section class="section">
    <h2>Curvas de entrenamiento</h2>
    <div class="plot-card full-plot">
        {BuildTrainingCurves(ProjectRoot, Summary)}
    </div>
</section>

<section class="section">
    <h2>Comparación visual sobre casos fijos</h2>
    <p class="section-intro">
        Se muestran pocos casos compartidos entre modelos para que la comparación sea legible.
        Cada imagen abre el PNG original en una pestaña nueva.
    </p>
    {BuildFixedCaseVisuals(ProjectRoot, Summary, MaxCases=MaxVisualCases)}
</section>

<section class="section soft-section">
    <h2>Detalles técnicos</h2>
    <p class="section-intro">
        Las tablas completas quedan plegadas. No se muestran por defecto para mantener el reporte limpio.
    </p>
    {BuildDetailsTables(Summary, BySample, LongBySample)}
</section>

<footer class="footer">
    Reporte generado automáticamente por Step16BuildComparisonHtmlReport.py · MethaneProjectTFM
</footer>

</div>
</body>
</html>
"""


def BuildCss() -> str:
    return """
    :root {
        --bg: #F5F9FF;
        --card: #FFFFFF;
        --ink: #16325C;
        --ink-dark: #0B2A63;
        --muted: #4F6787;
        --accent: #2F6FD6;
        --accent-2: #1C5DB8;
        --cyan: #5BB7E8;
        --line: rgba(22, 50, 92, 0.08);
        --shadow: 0 12px 30px rgba(31, 76, 155, 0.10);
        --shadow-soft: 0 6px 18px rgba(31, 76, 155, 0.06);
    }

    * { box-sizing: border-box; }

    body {
        margin: 0;
        padding: 28px;
        background: var(--bg);
        color: var(--ink);
        font-family: 'Montserrat', Arial, sans-serif;
        line-height: 1.58;
    }

    .container {
        max-width: 1580px;
        margin: 0 auto;
    }

    .hero {
        display: grid;
        grid-template-columns: 1.6fr 1fr;
        gap: 24px;
        align-items: end;
        background:
            radial-gradient(circle at 92% 10%, rgba(255,255,255,0.22), transparent 22%),
            linear-gradient(135deg, #0B2A63 0%, #1F65C8 52%, #2D7BE4 100%);
        color: white;
        border-radius: 26px;
        padding: 34px;
        margin-bottom: 24px;
        box-shadow: var(--shadow);
    }

    .eyebrow {
        color: rgba(255,255,255,0.72);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        line-height: 1.12;
        font-weight: 800;
        color: white;
    }

    .hero-text {
        margin: 14px 0 0 0;
        max-width: 820px;
        color: rgba(255,255,255,0.85);
    }

    .hero-meta {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }

    .hero-meta div {
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 16px;
        padding: 12px 14px;
        font-weight: 700;
    }

    .hero-meta span {
        display: block;
        font-size: 0.67rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.62);
        margin-bottom: 4px;
    }

    .section {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow-soft);
        padding: 26px;
        margin-bottom: 24px;
    }

    .soft-section {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
    }

    h2 {
        margin: 0 0 16px 0;
        color: var(--accent-2);
        font-size: 1.08rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    h3 {
        margin: 0 0 12px 0;
        color: var(--ink-dark);
        font-size: 1rem;
        font-weight: 800;
    }

    p {
        color: var(--muted);
        margin: 0 0 14px 0;
    }

    .muted {
        color: var(--muted);
        font-style: italic;
    }

    .section-intro {
        max-width: 1080px;
        font-size: 0.92rem;
    }

    .model-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
    }

    .model-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid rgba(47, 111, 214, 0.13);
        border-top: 6px solid var(--model-color);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 5px 14px rgba(31, 76, 155, 0.045);
    }

    .model-chip {
        display: inline-block;
        color: white;
        background: var(--model-color);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 12px;
    }

    .model-card h3 {
        font-size: 1.04rem;
        line-height: 1.25;
        min-height: 2.5rem;
    }

    .model-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 14px;
    }

    .model-metrics div {
        background: #F5F9FF;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 10px;
    }

    .model-metrics span {
        display: block;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .model-metrics strong {
        color: var(--ink-dark);
        font-size: 1.2rem;
    }

    .winner-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }

    .winner-item {
        background: #FBFDFF;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 14px;
    }

    .winner-item span {
        display: block;
        color: var(--accent);
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .winner-item strong {
        display: block;
        color: var(--ink-dark);
        margin-top: 6px;
        font-size: 0.95rem;
    }

    .winner-item em {
        display: block;
        color: var(--muted);
        margin-top: 4px;
        font-style: normal;
        font-weight: 700;
    }

    .two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
    }

    .plot-card,
    .image-card {
        background: #FBFDFF;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(31, 76, 155, 0.045);
        min-width: 0;
    }

    .full-plot {
        width: 100%;
    }

    .case-block {
        margin-top: 22px;
        padding-top: 20px;
        border-top: 1px solid var(--line);
    }

    .case-block:first-of-type {
        border-top: none;
        padding-top: 0;
    }

    .image-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
    }

    .image-card img {
        width: 100%;
        display: block;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: #EEF4FF;
    }

    figcaption {
        margin-top: 10px;
        color: var(--muted);
        font-size: 0.78rem;
        text-align: center;
    }

    .details-block {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: white;
        padding: 14px 16px;
        margin-top: 12px;
    }

    .details-block summary {
        cursor: pointer;
        color: var(--ink-dark);
        font-weight: 800;
    }

    .table-wrap {
        overflow-x: auto;
        margin-top: 10px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        background: #FBFDFF;
        border-radius: 14px;
        overflow: hidden;
        font-size: 0.78rem;
    }

    th {
        text-align: left;
        color: var(--ink);
        background: #EAF2FF;
        padding: 9px 10px;
        white-space: nowrap;
    }

    td {
        padding: 9px 10px;
        border-top: 1px solid rgba(22, 50, 92, 0.06);
        color: var(--muted);
        white-space: nowrap;
    }

    a {
        color: var(--accent-2);
        font-weight: 700;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.78rem;
        padding: 20px;
    }

    @media (max-width: 1350px) {
        .model-grid,
        .winner-grid,
        .image-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 1050px) {
        body { padding: 18px; }
        .hero { grid-template-columns: 1fr; }
        .two-col { grid-template-columns: 1fr; }
    }
    """


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Construye dashboard HTML comparativo.")
    Parser.add_argument("--ComparisonTag", required=True)
    Parser.add_argument("--MaxVisualCases", type=int, default=3)
    Args = Parser.parse_args()

    ProjectRoot = Path.cwd()
    ComparisonRoot = ProjectRoot / "Outputs" / "Comparisons" / Args.ComparisonTag

    TablesDir = ComparisonRoot / "Tables"
    ReportsDir = ComparisonRoot / "Reports"
    AuditDir = ComparisonRoot / "Audit"

    ReportsDir.mkdir(parents=True, exist_ok=True)
    AuditDir.mkdir(parents=True, exist_ok=True)

    Summary = SafeReadCsv(TablesDir / "ComparisonSummary.csv", Required=True)
    BySample = SafeReadCsv(TablesDir / "ComparisonBySample.csv", Required=False)
    LongBySample = SafeReadCsv(TablesDir / "AllMetricsBySampleLong.csv", Required=False)

    if len(Summary) > 3:
        raise ValueError(
            f"Este dashboard está diseñado para máximo 3 modelos. "
            f"Recibidos: {len(Summary)}. Crea comparaciones separadas."
        )

    PlotlySrc = CopyPlotlyJs(ReportsDir)

    Html = BuildHtml(
        ProjectRoot=ProjectRoot,
        ComparisonTag=Args.ComparisonTag,
        PlotlySrc=PlotlySrc,
        Summary=Summary,
        BySample=BySample,
        LongBySample=LongBySample,
        MaxVisualCases=Args.MaxVisualCases,
    )

    ReportPath = ReportsDir / "ComparisonReport.html"
    AuditPath = AuditDir / "ComparisonHtmlReportAudit.json"

    ReportPath.write_text(Html, encoding="utf-8")

    Audit = BuildAuditRecord(
        ScriptName="Step16BuildComparisonHtmlReport.py",
        RunTag="Comparison",
        Parameters=vars(Args),
        Inputs={
            "ComparisonSummary": str(TablesDir / "ComparisonSummary.csv"),
            "ComparisonBySample": str(TablesDir / "ComparisonBySample.csv"),
            "AllMetricsBySampleLong": str(TablesDir / "AllMetricsBySampleLong.csv"),
        },
        Outputs={
            "ComparisonReport": str(ReportPath),
            "PlotlyJs": str(ReportsDir / PlotlySrc),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "ComparisonTag": Args.ComparisonTag,
            "Runs": int(len(Summary)),
            "HtmlMode": "CompactVisualComparisonMax3",
        },
    )

    WriteJson(Audit, AuditPath)

    print("\n=== Compact comparison HTML report created ===")
    print("Report:", ReportPath)
    print("Runs:", len(Summary))
    print("URL:")
    print(f"http://localhost:8010/Outputs/Comparisons/{Args.ComparisonTag}/Reports/ComparisonReport.html?v=compact")


if __name__ == "__main__":
    Main()
