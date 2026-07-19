#!/usr/bin/env python3
"""
Step14BuildRunHtmlReport.py

Dashboard HTML visual para una ejecución específica.

Diseño:
- Montserrat en HTML y Plotly.
- Gráficas interactivas con paleta azul.
- Pocas tablas visibles.
- Tablas técnicas ocultas en <details>.
- Solo 3 imágenes de predicción visibles.
- No copia imágenes de predicción.
- No usa base64.
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

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


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


def BuildModelRunId(ModelName: str, RunName: str | None) -> str:
    if RunName is None or str(RunName).strip() == "":
        return ModelName
    return f"{ModelName}_{str(RunName).strip().replace(' ', '')}"


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
                "filename": "MethaneProjectTFM_plot",
                "height": 900,
                "width": 1400,
                "scale": 2,
            },
        },
    )


def ApplyLayout(Figure: go.Figure, Title: str, Height: int = 430) -> go.Figure:
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


def BuildKpiCards(TestSummary: pd.DataFrame, BestEpochSummary: pd.DataFrame) -> str:
    if TestSummary.empty:
        return "<p class='muted'>No hay métricas Test disponibles.</p>"

    Row = TestSummary.iloc[0]

    BestEpoch = "N/D"
    if not BestEpochSummary.empty and "BestEpoch" in BestEpochSummary.columns:
        BestEpoch = Escape(BestEpochSummary.iloc[0].get("BestEpoch", "N/D"))

    Specs = [
        ("Mean Dice", Row.get("MeanDice", ""), "Métrica media por muestra"),
        ("Mean IoU", Row.get("MeanIoU", ""), "Métrica media por muestra"),
        ("Global Dice", Row.get("GlobalDice", ""), "Agregado a nivel de píxel"),
        ("Global IoU", Row.get("GlobalIoU", ""), "Agregado a nivel de píxel"),
        ("Precision", Row.get("MeanPrecision", ""), "Media por muestra"),
        ("Recall", Row.get("MeanRecall", ""), "Media por muestra"),
        ("Samples", Row.get("Samples", ""), "Muestras evaluadas"),
        ("Best epoch", BestEpoch, "Mejor validación"),
    ]

    Cards = []
    for Label, Value, Note in Specs:
        if Label in {"Samples", "Best epoch"}:
            Display = Escape(Value)
        else:
            Display = FormatNumber(Value, 4)

        Cards.append(
            f"""
            <article class="kpi-card">
              <div class="kpi-label">{Escape(Label)}</div>
              <div class="kpi-value">{Display}</div>
              <div class="kpi-note">{Escape(Note)}</div>
            </article>
            """
        )

    return "<div class='kpi-grid'>" + "\n".join(Cards) + "</div>"


def BuildMetricBars(TestSummary: pd.DataFrame) -> str:
    if TestSummary.empty:
        return "<p class='muted'>No disponible.</p>"

    Row = TestSummary.iloc[0]

    Labels = ["Mean<br>Dice", "Mean<br>IoU", "Global<br>Dice", "Global<br>IoU", "Precision", "Recall"]
    Keys = ["MeanDice", "MeanIoU", "GlobalDice", "GlobalIoU", "MeanPrecision", "MeanRecall"]
    Values = [float(Row.get(Key, np.nan)) for Key in Keys]

    Colors = ["#B7D1ED", "#7194C2", "#2C7DE1", "#123A7A", "#8BD9E8", "#5BB7E8"]

    Figure = go.Figure(
        data=[
            go.Bar(
                x=Labels,
                y=Values,
                marker=dict(color=Colors, line=dict(color=PALETTE["navy"], width=1.3)),
                text=[FormatNumber(Value, 3) for Value in Values],
                textposition="outside",
                textfont=dict(family=FONT, size=15, color=PALETTE["text"]),
            )
        ]
    )

    Figure.update_yaxes(title="Valor de métrica", range=[0, 1])
    Figure.update_xaxes(title="")

    ApplyLayout(Figure, "Métricas principales en Test", Height=430)
    return PlotToHtml(Figure)


def BuildTrainingCurve(History: pd.DataFrame) -> str:
    if History.empty or "Epoch" not in History.columns:
        return "<p class='muted'>No disponible.</p>"

    Figure = go.Figure()

    Specs = [
        ("TrainMeanDice", "Train Dice", PALETTE["navy"], "dash"),
        ("ValidationMeanDice", "Val Dice", PALETTE["navy"], "solid"),
        ("TrainMeanIoU", "Train IoU", PALETTE["cyan"], "dash"),
        ("ValidationMeanIoU", "Val IoU", PALETTE["cyan"], "solid"),
    ]

    for Column, Label, Color, Dash in Specs:
        if Column in History.columns:
            Figure.add_trace(
                go.Scatter(
                    x=History["Epoch"],
                    y=History[Column],
                    mode="lines+markers",
                    name=Label,
                    line=dict(color=Color, width=4, dash=Dash),
                    marker=dict(size=8),
                )
            )

    if "ValidationMeanDice" in History.columns:
        BestIndex = History["ValidationMeanDice"].idxmax()
        Best = History.loc[BestIndex]
        Figure.add_trace(
            go.Scatter(
                x=[Best["Epoch"]],
                y=[Best["ValidationMeanDice"]],
                mode="markers+text",
                name="Best Val Dice",
                marker=dict(size=18, color=PALETTE["navy"]),
                text=[f"{Best['ValidationMeanDice']:.3f}<br>ep.{int(Best['Epoch'])}"],
                textposition="bottom right",
                textfont=dict(family=FONT, size=14, color=PALETTE["navy"]),
            )
        )

    Figure.update_xaxes(title="Época")
    Figure.update_yaxes(title="Valor de métrica", range=[0, 1])

    ApplyLayout(Figure, "Curvas de entrenamiento", Height=500)
    return PlotToHtml(Figure)


def BuildLossCurve(History: pd.DataFrame) -> str:
    if History.empty or "Epoch" not in History.columns:
        return "<p class='muted'>No disponible.</p>"

    Figure = go.Figure()

    if "TrainLoss" in History.columns:
        Figure.add_trace(
            go.Scatter(
                x=History["Epoch"],
                y=History["TrainLoss"],
                mode="lines+markers",
                name="Train Loss",
                line=dict(color=PALETTE["navy"], width=4),
                marker=dict(size=8),
            )
        )

    if "ValidationLoss" in History.columns:
        Figure.add_trace(
            go.Scatter(
                x=History["Epoch"],
                y=History["ValidationLoss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color=PALETTE["cyan"], width=4, dash="dash"),
                marker=dict(size=8),
            )
        )

    Figure.update_xaxes(title="Época")
    Figure.update_yaxes(title="Loss")

    ApplyLayout(Figure, "Evolución de la pérdida", Height=420)
    return PlotToHtml(Figure)


def BuildBoxplot(TestBySample: pd.DataFrame) -> str:
    if TestBySample.empty:
        return "<p class='muted'>No disponible.</p>"

    Columns = [Column for Column in ["Dice", "IoU", "Precision", "Recall"] if Column in TestBySample.columns]
    if not Columns:
        return "<p class='muted'>No disponible.</p>"

    Colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["cyan"], PALETTE["cyan2"]]

    Figure = go.Figure()
    for Column, Color in zip(Columns, Colors):
        Figure.add_trace(
            go.Box(
                y=TestBySample[Column],
                name=Column,
                marker_color=Color,
                boxmean=True,
            )
        )

    Figure.update_yaxes(title="Valor", range=[0, 1])
    ApplyLayout(Figure, "Distribución de métricas por muestra", Height=440)
    return PlotToHtml(Figure)


def BuildCorrelationHeatmap(TestBySample: pd.DataFrame) -> str:
    if TestBySample.empty:
        return "<p class='muted'>No disponible.</p>"

    Columns = [
        Column for Column in [
            "Dice",
            "IoU",
            "Precision",
            "Recall",
            "GroundTruthPixels",
            "PredictedPixels",
            "FalsePositivePixels",
            "FalseNegativePixels",
            "ProbabilityMean",
            "ProbabilityMax",
        ]
        if Column in TestBySample.columns
    ]

    if len(Columns) < 2:
        return "<p class='muted'>No disponible.</p>"

    Corr = TestBySample[Columns].corr(numeric_only=True).round(2)

    Figure = go.Figure(
        data=go.Heatmap(
            z=Corr.values,
            x=Corr.columns,
            y=Corr.columns,
            colorscale=[
                [0.0, "#F1F6FB"],
                [0.5, "#5BB7E8"],
                [1.0, "#0B2A63"],
            ],
            zmin=-1,
            zmax=1,
            text=Corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(family=FONT, size=11, color=PALETTE["text"]),
            colorbar=dict(title="Corr."),
        )
    )

    Figure.update_layout(yaxis_autorange="reversed")
    ApplyLayout(Figure, "Matriz de correlación", Height=620)
    return PlotToHtml(Figure)


def BuildPlumeScatter(TestBySample: pd.DataFrame) -> str:
    Required = {"GroundTruthPixels", "Dice", "IoU"}
    if TestBySample.empty or not Required.issubset(TestBySample.columns):
        return "<p class='muted'>No disponible.</p>"

    Figure = go.Figure()

    Figure.add_trace(
        go.Scatter(
            x=TestBySample["GroundTruthPixels"],
            y=TestBySample["Dice"],
            mode="markers",
            name="Dice",
            marker=dict(color=PALETTE["navy"], size=8, opacity=0.65),
            text=TestBySample["SampleId"] if "SampleId" in TestBySample.columns else None,
            hovertemplate="Sample: %{text}<br>GT pixels: %{x}<br>Dice: %{y:.4f}<extra></extra>",
        )
    )

    Figure.add_trace(
        go.Scatter(
            x=TestBySample["GroundTruthPixels"],
            y=TestBySample["IoU"],
            mode="markers",
            name="IoU",
            marker=dict(color=PALETTE["cyan"], size=8, opacity=0.65),
            text=TestBySample["SampleId"] if "SampleId" in TestBySample.columns else None,
            hovertemplate="Sample: %{text}<br>GT pixels: %{x}<br>IoU: %{y:.4f}<extra></extra>",
        )
    )

    Figure.update_xaxes(title="Ground truth plume pixels")
    Figure.update_yaxes(title="Valor de métrica", range=[0, 1])

    ApplyLayout(Figure, "Tamaño de pluma vs desempeño", Height=450)
    return PlotToHtml(Figure)


def BuildTopCasesTable(TestBySample: pd.DataFrame, Mode: str, MaxRows: int = 8) -> str:
    if TestBySample.empty or "Dice" not in TestBySample.columns:
        return "<p class='muted'>No disponible.</p>"

    if Mode == "best":
        Work = TestBySample.sort_values("Dice", ascending=False).head(MaxRows)
        Title = "Mejores casos"
    else:
        Work = TestBySample.sort_values("Dice", ascending=True).head(MaxRows)
        Title = "Casos más difíciles"

    Columns = ["SampleId", "Dice", "IoU", "Precision", "Recall", "GroundTruthPixels", "PredictedPixels"]
    Existing = [Column for Column in Columns if Column in Work.columns]
    Work = Work[Existing].copy()

    Header = "".join(f"<th>{Escape(Column)}</th>" for Column in Work.columns)

    Rows = []
    for _, Row in Work.iterrows():
        Cells = ""
        for Column in Work.columns:
            Value = Row[Column]
            Text = FormatNumber(Value, 4) if isinstance(Value, float) else Escape(Value)
            Cells += f"<td>{Text}</td>"
        Rows.append(f"<tr>{Cells}</tr>")

    return f"""
    <div class="mini-table-card">
        <h3>{Escape(Title)}</h3>
        <div class="table-wrap">
        <table class="compact-table">
            <thead><tr>{Header}</tr></thead>
            <tbody>{''.join(Rows)}</tbody>
        </table>
        </div>
    </div>
    """



def BuildErrorHistogram(TestBySample: pd.DataFrame) -> str:
    """Histograma de falsos positivos y falsos negativos."""
    if TestBySample.empty:
        return "<p class='muted'>No disponible.</p>"

    Columns = [Column for Column in ["FalsePositivePixels", "FalseNegativePixels"] if Column in TestBySample.columns]

    if not Columns:
        return "<p class='muted'>No disponible.</p>"

    Figure = go.Figure()

    if "FalsePositivePixels" in TestBySample.columns:
        Figure.add_trace(
            go.Histogram(
                x=TestBySample["FalsePositivePixels"],
                name="False positive pixels",
                marker_color=PALETTE["cyan"],
                opacity=0.72,
                nbinsx=35,
            )
        )

    if "FalseNegativePixels" in TestBySample.columns:
        Figure.add_trace(
            go.Histogram(
                x=TestBySample["FalseNegativePixels"],
                name="False negative pixels",
                marker_color=PALETTE["navy"],
                opacity=0.72,
                nbinsx=35,
            )
        )

    Figure.update_layout(barmode="overlay")
    Figure.update_xaxes(title="Pixels")
    Figure.update_yaxes(title="Número de muestras")

    ApplyLayout(Figure, "Distribución de errores FP/FN", Height=430)
    return PlotToHtml(Figure)


def BuildProbabilityDistribution(TestBySample: pd.DataFrame) -> str:
    """Distribución de probabilidades medias y máximas por muestra."""
    if TestBySample.empty:
        return "<p class='muted'>No disponible.</p>"

    Columns = [Column for Column in ["ProbabilityMean", "ProbabilityMax"] if Column in TestBySample.columns]

    if not Columns:
        return "<p class='muted'>No disponible.</p>"

    Figure = go.Figure()

    if "ProbabilityMean" in TestBySample.columns:
        Figure.add_trace(
            go.Histogram(
                x=TestBySample["ProbabilityMean"],
                name="Mean probability",
                marker_color=PALETTE["cyan"],
                opacity=0.75,
                nbinsx=35,
            )
        )

    if "ProbabilityMax" in TestBySample.columns:
        Figure.add_trace(
            go.Histogram(
                x=TestBySample["ProbabilityMax"],
                name="Max probability",
                marker_color=PALETTE["navy"],
                opacity=0.55,
                nbinsx=35,
            )
        )

    Figure.update_layout(barmode="overlay")
    Figure.update_xaxes(title="Probabilidad")
    Figure.update_yaxes(title="Número de muestras")

    ApplyLayout(Figure, "Distribución de probabilidad predicha", Height=430)
    return PlotToHtml(Figure)


def SelectPredictionExamplesByMode(
    FigureIndex: pd.DataFrame,
    Mode: str,
    MaxImages: int = 3,
    Seed: int = 42,
) -> pd.DataFrame:
    """
    Selecciona imágenes para mostrar en el reporte.

    Mode:
    - best: usa BestPredictions si existe; si no, mayor Dice.
    - worst: usa WorstPredictions si existe; si no, menor Dice.
    - random: selección reproducible de casos no repetidos.
    """
    if FigureIndex.empty or "FigurePath" not in FigureIndex.columns:
        return pd.DataFrame()

    Work = FigureIndex.copy()

    if "SampleId" in Work.columns:
        Work["SampleId"] = Work["SampleId"].astype(str)

    if Mode == "best":
        if "CaseGroup" in Work.columns:
            Candidate = Work[Work["CaseGroup"] == "BestPredictions"].copy()
        else:
            Candidate = pd.DataFrame()

        if Candidate.empty and "Dice" in Work.columns:
            Candidate = Work.sort_values("Dice", ascending=False).copy()

        if "Order" in Candidate.columns:
            Candidate = Candidate.sort_values("Order")

        return Candidate.head(MaxImages)

    if Mode == "worst":
        if "CaseGroup" in Work.columns:
            Candidate = Work[Work["CaseGroup"] == "WorstPredictions"].copy()
        else:
            Candidate = pd.DataFrame()

        if Candidate.empty and "Dice" in Work.columns:
            Candidate = Work.sort_values("Dice", ascending=True).copy()

        if "Order" in Candidate.columns:
            Candidate = Candidate.sort_values("Order")

        return Candidate.head(MaxImages)

    if Mode == "random":
        Candidate = Work.copy()

        if "CaseGroup" in Candidate.columns:
            Candidate = Candidate[
                ~Candidate["CaseGroup"].isin(["BestPredictions", "WorstPredictions"])
            ].copy()

        if Candidate.empty:
            Candidate = Work.copy()

        if len(Candidate) <= MaxImages:
            return Candidate.head(MaxImages)

        return Candidate.sample(n=MaxImages, random_state=Seed)

    raise ValueError(f"Mode no soportado: {Mode}")


def BuildPredictionExamplesByMode(
    ModelRoot: Path,
    FigureIndex: pd.DataFrame,
    Mode: str,
    Title: str,
    Description: str,
    MaxImages: int = 3,
) -> str:
    """Construye una sección visual con predicciones."""
    Examples = SelectPredictionExamplesByMode(
        FigureIndex=FigureIndex,
        Mode=Mode,
        MaxImages=MaxImages,
    )

    if Examples.empty:
        return f"""
        <div class="prediction-section">
            <h3>{Escape(Title)}</h3>
            <p class="muted">No hay figuras disponibles para esta sección.</p>
        </div>
        """

    Cards = []

    for _, Row in Examples.iterrows():
        SourceFigure = ModelRoot / str(Row["FigurePath"])

        if not SourceFigure.exists():
            continue

        Link = "../" + str(SourceFigure.relative_to(ModelRoot)).replace("\\", "/")
        Caption = (
            f"{Row.get('CaseGroup', Mode)} · "
            f"{Row.get('SampleId', '')} · "
            f"Dice={FormatNumber(Row.get('Dice', ''), 3)} · "
            f"IoU={FormatNumber(Row.get('IoU', ''), 3)}"
        )

        Cards.append(
            f"""
            <article class="image-card">
                <a href="{Escape(Link)}" target="_blank">
                    <img loading="lazy" src="{Escape(Link)}" alt="{Escape(Caption)}"/>
                </a>
                <figcaption>{Escape(Caption)}</figcaption>
            </article>
            """
        )

    if not Cards:
        return f"""
        <div class="prediction-section">
            <h3>{Escape(Title)}</h3>
            <p class="muted">No fue posible cargar las imágenes seleccionadas.</p>
        </div>
        """

    return f"""
    <div class="prediction-section">
        <h3>{Escape(Title)}</h3>
        <p class="section-intro">{Escape(Description)}</p>
        <div class="image-grid">
            {''.join(Cards)}
        </div>
    </div>
    """


def PickPredictionExamples(FigureIndex: pd.DataFrame, MaxImages: int = 3) -> pd.DataFrame:
    if FigureIndex.empty or "FigurePath" not in FigureIndex.columns:
        return pd.DataFrame()

    Work = FigureIndex.copy()

    Selected = []

    PreferredGroups = ["FixedComparisonCases", "BestPredictions", "WorstPredictions"]

    for Group in PreferredGroups:
        if "CaseGroup" in Work.columns:
            GroupRows = Work[Work["CaseGroup"] == Group].copy()
            if not GroupRows.empty:
                if "Order" in GroupRows.columns:
                    GroupRows = GroupRows.sort_values("Order")
                Selected.append(GroupRows.iloc[0])

    if len(Selected) < MaxImages:
        ExistingIds = {str(Row.get("SampleId", "")) for Row in Selected}
        Fill = Work[~Work["SampleId"].astype(str).isin(ExistingIds)].copy() if "SampleId" in Work.columns else Work
        if "Order" in Fill.columns:
            Fill = Fill.sort_values("Order")

        for _, Row in Fill.iterrows():
            Selected.append(Row)
            if len(Selected) >= MaxImages:
                break

    if not Selected:
        return pd.DataFrame()

    return pd.DataFrame(Selected).head(MaxImages)


def BuildPredictionExamples(ModelRoot: Path, FigureIndex: pd.DataFrame, MaxImages: int = 3) -> str:
    Examples = PickPredictionExamples(FigureIndex, MaxImages=MaxImages)

    if Examples.empty:
        return "<p class='muted'>No hay figuras de predicción disponibles. Ejecuta Step13VisualizePredictions.py.</p>"

    Cards = []

    for _, Row in Examples.iterrows():
        SourceFigure = ModelRoot / str(Row["FigurePath"])

        if not SourceFigure.exists():
            continue

        Link = "../" + str(SourceFigure.relative_to(ModelRoot)).replace("\\", "/")
        Caption = (
            f"{Row.get('CaseGroup', '')} · "
            f"{Row.get('SampleId', '')} · "
            f"Dice={FormatNumber(Row.get('Dice', ''), 3)} · "
            f"IoU={FormatNumber(Row.get('IoU', ''), 3)}"
        )

        Cards.append(
            f"""
            <article class="image-card">
                <a href="{Escape(Link)}" target="_blank">
                    <img loading="lazy" src="{Escape(Link)}" alt="{Escape(Caption)}"/>
                </a>
                <figcaption>{Escape(Caption)}</figcaption>
            </article>
            """
        )

    if not Cards:
        return "<p class='muted'>No fue posible cargar las imágenes seleccionadas.</p>"

    return "<div class='image-grid'>" + "\n".join(Cards) + "</div>"


def BuildPredictionLinks(ModelRoot: Path, FigureIndex: pd.DataFrame, MaxRows: int = 30) -> str:
    if FigureIndex.empty or "FigurePath" not in FigureIndex.columns:
        return "<p class='muted'>No hay índice de figuras.</p>"

    Work = FigureIndex.copy()
    if "CaseGroup" in Work.columns and "Order" in Work.columns:
        Work = Work.sort_values(["CaseGroup", "Order"])

    Work = Work.head(MaxRows).copy()

    Rows = []
    for _, Row in Work.iterrows():
        SourceFigure = ModelRoot / str(Row["FigurePath"])

        if SourceFigure.exists():
            Link = "../" + str(SourceFigure.relative_to(ModelRoot)).replace("\\", "/")
            FigureHtml = f"<a href='{Escape(Link)}' target='_blank'>abrir PNG</a>"
        else:
            FigureHtml = "missing"

        Rows.append(
            {
                "CaseGroup": Row.get("CaseGroup", ""),
                "SampleId": Row.get("SampleId", ""),
                "Dice": Row.get("Dice", ""),
                "IoU": Row.get("IoU", ""),
                "Figure": FigureHtml,
            }
        )

    Header = "".join(f"<th>{Escape(Column)}</th>" for Column in Rows[0].keys())
    Body = []

    for Row in Rows:
        Body.append("<tr>")
        for Key, Value in Row.items():
            if Key == "Figure":
                Body.append(f"<td>{Value}</td>")
            elif isinstance(Value, float):
                Body.append(f"<td>{FormatNumber(Value)}</td>")
            else:
                Body.append(f"<td>{Escape(Value)}</td>")
        Body.append("</tr>")

    return f"""
    <div class="table-wrap">
    <table class="compact-table">
        <thead><tr>{Header}</tr></thead>
        <tbody>{''.join(Body)}</tbody>
    </table>
    </div>
    """


def DetailsTable(Title: str, Table: pd.DataFrame, MaxRows: int = 30) -> str:
    if Table.empty:
        return ""

    Header = "".join(f"<th>{Escape(Column)}</th>" for Column in Table.columns)
    Rows = []

    for _, Row in Table.head(MaxRows).iterrows():
        Cells = ""
        for Column in Table.columns:
            Value = Row[Column]
            Text = FormatNumber(Value) if isinstance(Value, float) else Escape(Value)
            Cells += f"<td>{Text}</td>"
        Rows.append(f"<tr>{Cells}</tr>")

    return f"""
    <details class="details-block">
        <summary>{Escape(Title)}</summary>
        <div class="table-wrap">
        <table class="compact-table">
            <thead><tr>{Header}</tr></thead>
            <tbody>{''.join(Rows)}</tbody>
        </table>
        </div>
    </details>
    """


def BuildHtml(
    RunTag: str,
    FeatureConfig: str,
    ModelName: str,
    RunName: str,
    ModelRunId: str,
    ModelRoot: Path,
    PlotlySrc: str,
    TrainingHistory: pd.DataFrame,
    BestEpochSummary: pd.DataFrame,
    ModelRunSummary: pd.DataFrame,
    TestSummary: pd.DataFrame,
    TestBySample: pd.DataFrame,
    FigureIndex: pd.DataFrame,
    MaxPredictionLinks: int,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>MethaneProjectTFM · {Escape(ModelRunId)}</title>
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
        <p class="eyebrow">MethaneProjectTFM · segmentation experiment</p>
        <h1>{Escape(ModelRunId)}</h1>
        <p class="hero-text">
            Dashboard de entrenamiento, evaluación y predicciones.
            Las gráficas son interactivas y las imágenes se limitan a tres ejemplos representativos.
        </p>
    </div>
    <div class="hero-meta">
        <div><span>RunTag</span>{Escape(RunTag)}</div>
        <div><span>Config</span>{Escape(FeatureConfig)}</div>
        <div><span>Model</span>{Escape(ModelName)}</div>
        <div><span>RunName</span>{Escape(RunName)}</div>
    </div>
</section>

<section class="section">
    <h2>Resumen ejecutivo</h2>
    {BuildKpiCards(TestSummary, BestEpochSummary)}
</section>

<section class="section">
    <h2>Resultados principales</h2>
    <div class="two-col">
        <div class="plot-card">{BuildMetricBars(TestSummary)}</div>
        <div class="plot-card">{BuildBoxplot(TestBySample)}</div>
    </div>
</section>

<section class="section">
    <h2>Entrenamiento</h2>
    <div class="two-col">
        <div class="plot-card">{BuildTrainingCurve(TrainingHistory)}</div>
        <div class="plot-card">{BuildLossCurve(TrainingHistory)}</div>
    </div>
</section>

<section class="section">
    <h2>Análisis de comportamiento</h2>
    <div class="two-col">
        <div class="plot-card">{BuildCorrelationHeatmap(TestBySample)}</div>
        <div class="plot-card">{BuildPlumeScatter(TestBySample)}</div>
    </div>
    <div class="two-col extra-plot-row">
        <div class="plot-card">{BuildErrorHistogram(TestBySample)}</div>
        <div class="plot-card">{BuildProbabilityDistribution(TestBySample)}</div>
    </div>
</section>

<section class="section">
    <h2>Ejemplos visuales</h2>
    <p class="section-intro">
        Se muestran nueve predicciones organizadas en tres grupos:
        tres mejores, tres peores y tres aleatorias reproducibles.
    </p>

    {BuildPredictionExamplesByMode(
        ModelRoot,
        FigureIndex,
        Mode="best",
        Title="3 mejores predicciones",
        Description="Casos con mejor desempeño visual y mayor Dice dentro de las figuras generadas.",
        MaxImages=3,
    )}

    {BuildPredictionExamplesByMode(
        ModelRoot,
        FigureIndex,
        Mode="worst",
        Title="3 peores predicciones",
        Description="Casos difíciles donde el modelo tiene menor Dice o mayor error espacial.",
        MaxImages=3,
    )}

    {BuildPredictionExamplesByMode(
        ModelRoot,
        FigureIndex,
        Mode="random",
        Title="3 casos aleatorios",
        Description="Casos seleccionados de forma reproducible para inspección general.",
        MaxImages=3,
    )}
</section>

<section class="section">
    <h2>Casos destacados</h2>
    <div class="two-col">
        {BuildTopCasesTable(TestBySample, Mode="best", MaxRows=8)}
        {BuildTopCasesTable(TestBySample, Mode="worst", MaxRows=8)}
    </div>
</section>

<section class="section soft-section">
    <h2>Detalles técnicos</h2>
    <p class="section-intro">
        Las tablas completas se dejan plegadas para no saturar la lectura del reporte.
    </p>
    {DetailsTable("ModelRunSummary", ModelRunSummary, MaxRows=1)}
    {DetailsTable("BestEpochSummary", BestEpochSummary, MaxRows=1)}
    {DetailsTable("TrainingHistory", TrainingHistory, MaxRows=80)}
    {DetailsTable("TestMetricsSummary", TestSummary, MaxRows=1)}
    <details class="details-block">
        <summary>Links a figuras de predicción</summary>
        {BuildPredictionLinks(ModelRoot, FigureIndex, MaxRows=MaxPredictionLinks)}
    </details>
</section>

<footer class="footer">
    Reporte generado automáticamente por Step14BuildRunHtmlReport.py · MethaneProjectTFM
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
        --cyan-soft: #E9F7FF;
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
        font-size: 2.25rem;
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

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(8, minmax(120px, 1fr));
        gap: 12px;
    }

    .kpi-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid rgba(47, 111, 214, 0.13);
        border-radius: 18px;
        padding: 16px 14px;
        min-height: 112px;
        box-shadow: 0 5px 14px rgba(31, 76, 155, 0.045);
    }

    .kpi-label {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }

    .kpi-value {
        color: var(--ink-dark);
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .kpi-note {
        color: var(--muted);
        font-size: 0.72rem;
    }

    .two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
    }

    .plot-card,
    .mini-table-card,
    .image-card {
        background: #FBFDFF;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(31, 76, 155, 0.045);
        min-width: 0;
    }

    .extra-plot-row {
        margin-top: 18px;
    }

    .prediction-section {
        margin-top: 26px;
        padding-top: 18px;
        border-top: 1px solid var(--line);
    }

    .prediction-section:first-of-type {
        border-top: none;
        padding-top: 0;
    }

    .prediction-section h3 {
        color: var(--ink-dark);
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .extra-plot-row {
        margin-top: 18px;
    }

    .prediction-section {
        margin-top: 26px;
        padding-top: 18px;
        border-top: 1px solid var(--line);
    }

    .prediction-section:first-of-type {
        border-top: none;
        padding-top: 0;
    }

    .prediction-section h3 {
        color: var(--ink-dark);
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .extra-plot-row {
        margin-top: 18px;
    }

    .prediction-section {
        margin-top: 26px;
        padding-top: 18px;
        border-top: 1px solid var(--line);
    }

    .prediction-section:first-of-type {
        border-top: none;
        padding-top: 0;
    }

    .prediction-section h3 {
        color: var(--ink-dark);
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .image-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        margin-top: 18px;
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
        font-size: 0.8rem;
    }

    table.compact-table {
        font-size: 0.76rem;
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
        .kpi-grid { grid-template-columns: repeat(4, 1fr); }
        .image-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 1050px) {
        body { padding: 18px; }
        .hero { grid-template-columns: 1fr; }
        .two-col { grid-template-columns: 1fr; }
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }

    @media (max-width: 650px) {
        .kpi-grid { grid-template-columns: 1fr; }
        .hero-meta { grid-template-columns: 1fr; }
    }
    """


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Construye dashboard HTML visual de una ejecución.")
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--FeatureConfig", required=True, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--ModelName", required=True)
    Parser.add_argument("--RunName", required=True)
    Parser.add_argument("--MaxPredictionLinks", type=int, default=30)

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    ModelRunId = BuildModelRunId(Args.ModelName, Args.RunName)

    ModelRoot = Paths.RunDirectory / Args.FeatureConfig / ModelRunId
    MetricsDirectory = ModelRoot / "Metrics"
    TablesDirectory = ModelRoot / "Tables"
    ReportsDirectory = ModelRoot / "Reports"
    AuditDirectory = ModelRoot / "Audit"

    ReportsDirectory.mkdir(parents=True, exist_ok=True)
    AuditDirectory.mkdir(parents=True, exist_ok=True)

    PlotlySrc = CopyPlotlyJs(ReportsDirectory)

    TrainingHistory = SafeReadCsv(MetricsDirectory / "TrainingHistory.csv", Required=True)
    BestEpochSummary = SafeReadCsv(MetricsDirectory / "BestEpochSummary.csv")
    ModelRunSummary = SafeReadCsv(TablesDirectory / "ModelRunSummary.csv")
    TestSummary = SafeReadCsv(MetricsDirectory / "TestMetricsSummary.csv")
    TestBySample = SafeReadCsv(MetricsDirectory / "TestMetricsBySample.csv")
    FigureIndex = SafeReadCsv(TablesDirectory / "PredictionFigureIndex.csv")

    Html = BuildHtml(
        RunTag=Args.RunTag,
        FeatureConfig=Args.FeatureConfig,
        ModelName=Args.ModelName,
        RunName=Args.RunName,
        ModelRunId=ModelRunId,
        ModelRoot=ModelRoot,
        PlotlySrc=PlotlySrc,
        TrainingHistory=TrainingHistory,
        BestEpochSummary=BestEpochSummary,
        ModelRunSummary=ModelRunSummary,
        TestSummary=TestSummary,
        TestBySample=TestBySample,
        FigureIndex=FigureIndex,
        MaxPredictionLinks=Args.MaxPredictionLinks,
    )

    ReportPath = ReportsDirectory / "RunReport.html"
    AuditPath = AuditDirectory / "RunHtmlReportAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    ReportPath.write_text(Html, encoding="utf-8")

    Audit = BuildAuditRecord(
        ScriptName="Step14BuildRunHtmlReport.py",
        RunTag=Args.RunTag,
        Parameters=vars(Args),
        Inputs={
            "TrainingHistory": str(MetricsDirectory / "TrainingHistory.csv"),
            "BestEpochSummary": str(MetricsDirectory / "BestEpochSummary.csv"),
            "ModelRunSummary": str(TablesDirectory / "ModelRunSummary.csv"),
            "TestMetricsSummary": str(MetricsDirectory / "TestMetricsSummary.csv"),
            "TestMetricsBySample": str(MetricsDirectory / "TestMetricsBySample.csv"),
            "PredictionFigureIndex": str(TablesDirectory / "PredictionFigureIndex.csv"),
        },
        Outputs={
            "RunReport": str(ReportPath),
            "PlotlyJs": str(ReportsDirectory / PlotlySrc),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "ModelRunId": ModelRunId,
            "HtmlMode": "VisualPlotlyDashboardMontserratThreeImages",
            "HtmlSizeBytes": int(ReportPath.stat().st_size),
            "VisiblePredictionImages": 3,
        },
    )

    WriteJson(Audit, AuditPath)

    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=Args.RunTag,
        Step="Step14BuildRunHtmlReport",
        Config=Args.FeatureConfig,
        Model=ModelRunId,
        OutputType="Report",
        RelativePath=str(ReportPath.relative_to(Paths.RunDirectory)),
        Created=ReportPath.exists(),
        Description=f"Dashboard HTML visual con Montserrat y 3 imágenes para {ModelRunId}.",
    )

    print("\n=== Visual Plotly Run HTML dashboard created ===")
    print("Report:", ReportPath)
    print("Plotly JS:", ReportsDirectory / PlotlySrc)
    print("Size bytes:", ReportPath.stat().st_size)
    print("Visible prediction images: 3")


if __name__ == "__main__":
    Main()
