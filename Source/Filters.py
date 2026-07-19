"""
Filters.py

Filtros experimentales para MethaneProjectTFM.

Filtros esperados:
- detection:sector == Oil and Gas
- quality:percentage_clear >= 90.0
- quality:observability == clear
- detection:isplume == True
- plume:geometry no nulo
- presencia de target, reference y plume

Cada filtro genera una fila en FilterSummary.csv.
"""

from __future__ import annotations

import pandas as pd


FILTER_COLUMN_CANDIDATES = {
    "Sector": ["detection:sector", "sector", "Sector"],
    "PercentageClear": ["quality:percentage_clear", "percentage_clear", "clear_percentage"],
    "Observability": ["quality:observability", "observability"],
    "IsPlume": ["detection:isplume", "isplume", "is_plume"],
    "PlumeGeometry": ["plume:geometry", "geometry", "plume_geometry"],
}


def FindColumn(DataFrame: pd.DataFrame, Candidates: list[str]) -> str | None:
    """Busca la primera columna existente entre candidatas."""
    for Column in Candidates:
        if Column in DataFrame.columns:
            return Column
    return None


def DetectFilterColumns(DataFrame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Detecta columnas necesarias para filtros."""
    Detected = {}
    Rows = []

    for Concept, Candidates in FILTER_COLUMN_CANDIDATES.items():
        Column = FindColumn(DataFrame, Candidates)
        Detected[Concept] = Column
        Rows.append(
            {
                "Concept": Concept,
                "Column": Column or "",
                "Exists": Column is not None,
                "Candidates": "|".join(Candidates),
            }
        )

    for ProductFlag in ["HasTarget", "HasReference", "HasPlume", "HasCH4", "HasDem"]:
        Detected[ProductFlag] = ProductFlag if ProductFlag in DataFrame.columns else None
        Rows.append(
            {
                "Concept": ProductFlag,
                "Column": ProductFlag if ProductFlag in DataFrame.columns else "",
                "Exists": ProductFlag in DataFrame.columns,
                "Candidates": ProductFlag,
            }
        )

    return pd.DataFrame(Rows), Detected


def NormalizeText(Series: pd.Series) -> pd.Series:
    """Normaliza textos para filtros."""
    return Series.astype(str).str.strip().str.lower()


def AddFilterSummaryRow(
    Rows: list[dict],
    FilterName: str,
    Column: str | None,
    Before: int,
    After: int,
    Applied: bool,
    Criterion: str,
) -> None:
    """Añade fila de resumen de filtro."""
    Rows.append(
        {
            "Filter": FilterName,
            "Column": Column or "",
            "Before": int(Before),
            "After": int(After),
            "Removed": int(Before - After),
            "Applied": bool(Applied),
            "Criterion": Criterion,
        }
    )


def ApplyMandatoryFilters(
    DatasetIndex: pd.DataFrame,
    ProjectConfig: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica filtros obligatorios definidos en ProjectConfig."""
    DataFrame = DatasetIndex.copy()
    DetectedColumnsTable, DetectedColumns = DetectFilterColumns(DataFrame)
    SummaryRows = []

    AddFilterSummaryRow(
        SummaryRows,
        "InitialSamples",
        None,
        len(DataFrame),
        len(DataFrame),
        True,
        "DatasetIndex inicial",
    )

    FiltersConfig = ProjectConfig["Filters"]

    SectorColumn = DetectedColumns["Sector"]
    if SectorColumn is None:
        raise KeyError("No se encontró columna para filtro Sector.")

    Before = len(DataFrame)
    ExpectedSector = str(FiltersConfig["Sector"]).strip().lower()
    DataFrame = DataFrame[NormalizeText(DataFrame[SectorColumn]) == ExpectedSector].copy()
    AddFilterSummaryRow(
        SummaryRows,
        "Sector",
        SectorColumn,
        Before,
        len(DataFrame),
        True,
        f"== {FiltersConfig['Sector']}",
    )

    ClearColumn = DetectedColumns["PercentageClear"]
    if ClearColumn is None:
        raise KeyError("No se encontró columna para filtro PercentageClear.")

    Before = len(DataFrame)
    MinPercentageClear = float(FiltersConfig["MinPercentageClear"])
    ClearValues = pd.to_numeric(DataFrame[ClearColumn], errors="coerce")
    DataFrame = DataFrame[ClearValues >= MinPercentageClear].copy()
    AddFilterSummaryRow(
        SummaryRows,
        "PercentageClear",
        ClearColumn,
        Before,
        len(DataFrame),
        True,
        f">= {MinPercentageClear}",
    )

    ObservabilityColumn = DetectedColumns["Observability"]
    if ObservabilityColumn is None:
        raise KeyError("No se encontró columna para filtro Observability.")

    Before = len(DataFrame)
    ExpectedObservability = str(FiltersConfig["Observability"]).strip().lower()
    DataFrame = DataFrame[
        NormalizeText(DataFrame[ObservabilityColumn]) == ExpectedObservability
    ].copy()
    AddFilterSummaryRow(
        SummaryRows,
        "Observability",
        ObservabilityColumn,
        Before,
        len(DataFrame),
        True,
        f"== {FiltersConfig['Observability']}",
    )

    IsPlumeColumn = DetectedColumns["IsPlume"]
    if bool(FiltersConfig.get("RequirePlume", True)):
        if IsPlumeColumn is None:
            raise KeyError("No se encontró columna para filtro IsPlume.")

        Before = len(DataFrame)
        DataFrame = DataFrame[DataFrame[IsPlumeColumn].astype(bool)].copy()
        AddFilterSummaryRow(
            SummaryRows,
            "IsPlume",
            IsPlumeColumn,
            Before,
            len(DataFrame),
            True,
            "== True",
        )

    GeometryColumn = DetectedColumns["PlumeGeometry"]
    if bool(FiltersConfig.get("RequireGeometry", True)):
        if GeometryColumn is None:
            raise KeyError("No se encontró columna para filtro PlumeGeometry.")

        Before = len(DataFrame)
        DataFrame = DataFrame[DataFrame[GeometryColumn].notna()].copy()
        AddFilterSummaryRow(
            SummaryRows,
            "PlumeGeometry",
            GeometryColumn,
            Before,
            len(DataFrame),
            True,
            "not null",
        )

    RequiredProducts = FiltersConfig.get("RequireProducts", ["target", "reference", "plume"])
    ProductFlagMap = {
        "target": "HasTarget",
        "reference": "HasReference",
        "plume": "HasPlume",
        "ch4": "HasCH4",
        "dem": "HasDem",
    }

    for ProductName in RequiredProducts:
        FlagColumn = ProductFlagMap.get(ProductName)
        if FlagColumn is None:
            raise ValueError(f"Producto requerido no reconocido: {ProductName}")

        if FlagColumn not in DataFrame.columns:
            raise KeyError(f"No existe columna de presencia de producto: {FlagColumn}")

        Before = len(DataFrame)
        DataFrame = DataFrame[DataFrame[FlagColumn].astype(bool)].copy()
        AddFilterSummaryRow(
            SummaryRows,
            FlagColumn,
            FlagColumn,
            Before,
            len(DataFrame),
            True,
            f"{ProductName} present",
        )

    DataFrame = DataFrame.reset_index(drop=True)
    FilterSummary = pd.DataFrame(SummaryRows)

    if DataFrame.empty:
        raise ValueError("Los filtros dejaron el dataset vacío. Revisa FilterSummary.csv.")

    return DataFrame, FilterSummary, DetectedColumnsTable
