"""
HtmlReportUtils.py

Utilidades comunes para reportes HTML de MethaneProjectTFM.

La estética sigue la línea visual del póster:
- Montserrat para títulos;
- Comfortaa para cuerpo;
- paleta azul;
- tarjetas blancas;
- bordes suaves;
- sombras ligeras;
- reportes autocontenidos con imágenes embebidas en base64 cuando sea posible.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import pandas as pd


def ReadTextSafe(PathItem: Path) -> str:
    """Lee texto si existe; devuelve vacío si no."""
    if not PathItem.exists():
        return ""
    return PathItem.read_text(encoding="utf-8")


def ImageToBase64(PathItem: Path) -> str:
    """Convierte imagen a data URI base64."""
    if not PathItem.exists():
        return ""

    Suffix = PathItem.suffix.lower().replace(".", "")
    Mime = "image/png" if Suffix == "png" else f"image/{Suffix}"

    Data = base64.b64encode(PathItem.read_bytes()).decode("utf-8")
    return f"data:{Mime};base64,{Data}"


def Escape(Value: Any) -> str:
    """Escapa valores para HTML."""
    if pd.isna(Value):
        return ""
    return html.escape(str(Value))


def FormatFloat(Value: Any, Decimals: int = 4) -> str:
    """Formatea floats de forma segura."""
    try:
        return f"{float(Value):.{Decimals}f}"
    except Exception:
        return Escape(Value)


def DataFrameToHtmlTable(
    Table: pd.DataFrame,
    Columns: list[str] | None = None,
    MaxRows: int | None = None,
    CssClass: str = "data-table",
) -> str:
    """Convierte DataFrame a tabla HTML compacta."""
    if Table is None or Table.empty:
        return '<div class="empty">No hay datos disponibles.</div>'

    Work = Table.copy()

    if Columns is not None:
        Existing = [Column for Column in Columns if Column in Work.columns]
        Work = Work[Existing]

    if MaxRows is not None:
        Work = Work.head(MaxRows)

    Header = "".join(f"<th>{Escape(Column)}</th>" for Column in Work.columns)

    Rows = []
    for _, Row in Work.iterrows():
        Cells = "".join(f"<td>{Escape(Row[Column])}</td>" for Column in Work.columns)
        Rows.append(f"<tr>{Cells}</tr>")

    Body = "\n".join(Rows)

    return f"""
<table class="{CssClass}">
<thead><tr>{Header}</tr></thead>
<tbody>
{Body}
</tbody>
</table>
"""


def BuildReportCss() -> str:
    """CSS común para reportes HTML."""
    return """
<link href="https://fonts.googleapis.com" rel="preconnect"/>
<link crossorigin="" href="https://fonts.gstatic.com" rel="preconnect"/>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Comfortaa:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bd:#0A2F6E;
  --bk:#1246A8;
  --bm:#1E7EE6;
  --bl:#4FACF5;
  --bp:#C7DCF0;
  --bg:#EBF3FB;
  --bgg:#D6EAFF;
  --tx:#1B2B4B;
  --tm:#4A6080;
  --wh:#fff;
  --ok:#2ca02c;
  --warn:#ff7f0e;
  --bad:#d62728;
  --sh:0 2px 12px rgba(10,47,110,0.10);
  --br:12px;
  --cb:1px solid #C7DCF0;
  --g:14px;
}
body{
  background:var(--bg);
  font-family:'Comfortaa',sans-serif;
  color:var(--tx);
  line-height:1.55;
}
.report{
  width:min(1280px, calc(100vw - 32px));
  margin:0 auto;
  padding:22px 0 40px;
}
.header{
  background:linear-gradient(125deg,#051840 0%,#0D3582 52%,#1E7EE6 100%);
  border-radius:var(--br);
  padding:22px 28px;
  color:#fff;
  box-shadow:var(--sh);
  margin-bottom:var(--g);
}
.header h1{
  font-family:'Montserrat',sans-serif;
  font-size:28px;
  line-height:1.2;
  font-weight:800;
  margin-bottom:8px;
}
.header .accent{color:#9AD8FF;}
.header-meta{
  display:flex;
  flex-wrap:wrap;
  gap:18px;
  color:rgba(255,255,255,.84);
  font-size:13px;
}
.header-meta strong{
  font-family:'Montserrat',sans-serif;
  font-weight:700;
  color:rgba(255,255,255,.55);
  text-transform:uppercase;
  letter-spacing:.5px;
  margin-right:4px;
}
.grid{
  display:grid;
  grid-template-columns:repeat(12,1fr);
  gap:var(--g);
  margin-bottom:var(--g);
}
.card{
  background:var(--wh);
  border:var(--cb);
  border-radius:var(--br);
  box-shadow:var(--sh);
  padding:18px 20px;
}
.card.span-12{grid-column:span 12;}
.card.span-8{grid-column:span 8;}
.card.span-6{grid-column:span 6;}
.card.span-4{grid-column:span 4;}
.card.span-3{grid-column:span 3;}
.section-title{
  font-family:'Montserrat',sans-serif;
  font-size:14px;
  font-weight:800;
  letter-spacing:.8px;
  text-transform:uppercase;
  color:var(--bk);
  margin-bottom:12px;
  display:flex;
  align-items:center;
  gap:8px;
}
.section-title::before{
  content:'';
  width:16px;
  height:4px;
  background:var(--bm);
  border-radius:4px;
  display:inline-block;
}
.note{
  background:var(--bg);
  border:var(--cb);
  border-left:4px solid var(--bm);
  border-radius:10px;
  padding:12px 14px;
  color:var(--tx);
  font-size:13px;
}
.kpis{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:10px;
}
.kpi{
  background:linear-gradient(135deg,var(--bk),var(--bm));
  color:#fff;
  border-radius:12px;
  padding:14px 12px;
  text-align:center;
}
.kpi.alt{background:linear-gradient(135deg,#0A2F6E,#4FACF5);}
.kpi-value{
  display:block;
  font-family:'Montserrat',sans-serif;
  font-size:26px;
  font-weight:800;
  line-height:1;
}
.kpi-label{
  display:block;
  margin-top:6px;
  font-family:'Montserrat',sans-serif;
  font-size:11px;
  color:rgba(255,255,255,.78);
}
.data-table{
  width:100%;
  border-collapse:collapse;
  font-size:12px;
}
.data-table th{
  font-family:'Montserrat',sans-serif;
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:.4px;
  color:var(--tm);
  background:var(--bg);
  padding:8px 9px;
  border-bottom:2px solid var(--bp);
  text-align:left;
}
.data-table td{
  padding:8px 9px;
  border-bottom:1px solid var(--bg);
}
.data-table tr:hover td{background:#F6FAFF;}
.figure{
  width:100%;
  border-radius:10px;
  border:var(--cb);
  background:#fff;
  display:block;
}
.figure-grid{
  display:grid;
  grid-template-columns:repeat(2,1fr);
  gap:12px;
}
.figure-grid.three{
  grid-template-columns:repeat(3,1fr);
}
.fig-card{
  background:#fff;
  border:var(--cb);
  border-radius:10px;
  padding:10px;
}
.fig-card img{
  width:100%;
  border-radius:8px;
  display:block;
}
.fig-caption{
  margin-top:7px;
  font-size:11px;
  color:var(--tm);
  font-family:'Montserrat',sans-serif;
}
.badge{
  display:inline-block;
  padding:3px 8px;
  border-radius:999px;
  background:var(--bgg);
  color:var(--bd);
  font-family:'Montserrat',sans-serif;
  font-size:11px;
  font-weight:700;
}
.empty{
  background:var(--bg);
  border:var(--cb);
  border-radius:10px;
  padding:12px;
  color:var(--tm);
  font-style:italic;
}
.footer{
  margin-top:20px;
  color:var(--tm);
  font-size:11px;
  text-align:center;
}
@media(max-width:900px){
  .card.span-8,.card.span-6,.card.span-4,.card.span-3{grid-column:span 12;}
  .kpis{grid-template-columns:repeat(2,1fr);}
  .figure-grid,.figure-grid.three{grid-template-columns:1fr;}
}
</style>
"""


def BuildHtmlDocument(Title: str, Body: str) -> str:
    """Construye documento HTML completo."""
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{Escape(Title)}</title>
{BuildReportCss()}
</head>
<body>
<div class="report">
{Body}
</div>
</body>
</html>
"""
