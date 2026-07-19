"""
TacoIndex.py

Lectura e indexación del dataset TACO.

Responsabilidades:
- cargar dataset TACO con tacoreader;
- extraer tabla principal ds.data;
- extraer tabla plana ds.flatten();
- construir presencia de productos por muestra;
- unir metadatos de muestra con disponibilidad de productos.

Este módulo no aplica filtros ni crea splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def TacoObjectToDataFrame(TacoObject: Any, ObjectName: str) -> pd.DataFrame:
    """Convierte objetos devueltos por tacoreader a pandas.DataFrame."""
    if isinstance(TacoObject, pd.DataFrame):
        return TacoObject.copy()

    CandidateMethods = ["to_pandas", "to_dataframe", "to_df", "compute"]

    for MethodName in CandidateMethods:
        if hasattr(TacoObject, MethodName):
            Result = getattr(TacoObject, MethodName)()
            if isinstance(Result, pd.DataFrame):
                return Result.copy()

    CandidateAttributes = ["data", "df", "_data", "_df"]

    for AttributeName in CandidateAttributes:
        if hasattr(TacoObject, AttributeName):
            Result = getattr(TacoObject, AttributeName)
            if isinstance(Result, pd.DataFrame):
                return Result.copy()

    try:
        DataFrame = pd.DataFrame(TacoObject)
        if not DataFrame.empty:
            return DataFrame
    except Exception as Error:
        raise TypeError(
            f"No fue posible convertir {ObjectName} a pandas.DataFrame. "
            f"Tipo recibido: {type(TacoObject)}"
        ) from Error

    raise TypeError(
        f"No fue posible convertir {ObjectName} a pandas.DataFrame. "
        f"Tipo recibido: {type(TacoObject)}"
    )


def LoadTacoDataset(DataRoot: str | Path, DatasetName: str):
    """Carga dataset TACO desde DataRoot/DatasetName usando tacoreader."""
    DatasetPath = Path(DataRoot) / DatasetName

    if not DatasetPath.exists():
        raise FileNotFoundError(f"No existe la ruta del dataset TACO: {DatasetPath}")

    try:
        import tacoreader
    except ImportError as Error:
        raise ImportError(
            "No se pudo importar tacoreader en el entorno actual. "
            "Activa el entorno correcto o instala tacoreader antes de ejecutar esta fase."
        ) from Error

    if hasattr(tacoreader, "use"):
        tacoreader.use("pandas")

    Dataset = tacoreader.load(str(DatasetPath))

    DatasetInfo = {
        "DataRoot": str(DataRoot),
        "DatasetName": DatasetName,
        "DatasetPath": str(DatasetPath),
    }

    return Dataset, DatasetInfo


def GetSampleTable(Dataset) -> pd.DataFrame:
    """Extrae la tabla principal ds.data."""
    if not hasattr(Dataset, "data"):
        raise AttributeError("El dataset TACO no tiene atributo .data")

    SampleTable = TacoObjectToDataFrame(Dataset.data, "Dataset.data")

    if SampleTable.empty:
        raise ValueError("Dataset.data está vacío.")

    return SampleTable


def GetFlattenedTable(Dataset) -> pd.DataFrame:
    """Extrae la tabla plana ds.flatten()."""
    if not hasattr(Dataset, "flatten"):
        raise AttributeError("El dataset TACO no tiene método .flatten().")

    FlattenedTable = TacoObjectToDataFrame(Dataset.flatten(), "Dataset.flatten()")

    if FlattenedTable.empty:
        raise ValueError("Dataset.flatten() está vacío.")

    return FlattenedTable


def DetectColumn(DataFrame: pd.DataFrame, Candidates: list[str], Required: bool = True) -> str | None:
    """Detecta una columna entre candidatas."""
    for Column in Candidates:
        if Column in DataFrame.columns:
            return Column

    if Required:
        raise KeyError(f"No se encontró ninguna columna candidata: {Candidates}")

    return None


def BuildProductIndex(FlattenedTable: pd.DataFrame) -> pd.DataFrame:
    """Construye índice producto-muestra desde la tabla plana."""
    SampleIdColumn = DetectColumn(FlattenedTable, ["l0:id", "sample_id", "id"])
    ProductIdColumn = DetectColumn(FlattenedTable, ["l1:id", "product_id", "asset_id"])
    ProductTypeColumn = DetectColumn(
        FlattenedTable,
        ["l1:type", "product_type", "asset_type"],
        Required=False,
    )

    Columns = [SampleIdColumn, ProductIdColumn]
    if ProductTypeColumn is not None:
        Columns.append(ProductTypeColumn)

    ProductIndex = FlattenedTable[Columns].drop_duplicates().copy()
    ProductIndex = ProductIndex.rename(
        columns={
            SampleIdColumn: "SampleId",
            ProductIdColumn: "ProductId",
            ProductTypeColumn: "ProductType" if ProductTypeColumn is not None else ProductIdColumn,
        }
    )

    ProductIndex = ProductIndex.sort_values(["SampleId", "ProductId"]).reset_index(drop=True)

    return ProductIndex


def BuildProductPresence(ProductIndex: pd.DataFrame) -> pd.DataFrame:
    """Resume presencia de productos por muestra."""
    RequiredColumns = ["SampleId", "ProductId"]
    MissingColumns = [Column for Column in RequiredColumns if Column not in ProductIndex.columns]

    if MissingColumns:
        raise KeyError(f"ProductIndex no contiene columnas requeridas: {MissingColumns}")

    ProductPresence = (
        ProductIndex.groupby("SampleId")["ProductId"]
        .apply(lambda Values: sorted(set(str(Value) for Value in Values.dropna())))
        .reset_index(name="Products")
    )

    ProductPresence["HasTarget"] = ProductPresence["Products"].apply(lambda Items: "target" in Items)
    ProductPresence["HasReference"] = ProductPresence["Products"].apply(
        lambda Items: "reference" in Items
    )
    ProductPresence["HasPlume"] = ProductPresence["Products"].apply(lambda Items: "plume" in Items)
    ProductPresence["HasCH4"] = ProductPresence["Products"].apply(lambda Items: "ch4" in Items)
    ProductPresence["HasDem"] = ProductPresence["Products"].apply(lambda Items: "dem" in Items)
    ProductPresence["ProductCount"] = ProductPresence["Products"].apply(len)

    ProductPresence["Products"] = ProductPresence["Products"].apply(lambda Items: "|".join(Items))

    return ProductPresence


def BuildDatasetIndex(SampleTable: pd.DataFrame, ProductPresence: pd.DataFrame) -> pd.DataFrame:
    """Une tabla principal de muestras con presencia de productos."""
    SampleIdColumn = DetectColumn(SampleTable, ["id", "SampleId", "sample_id"])

    DatasetIndex = SampleTable.copy()
    DatasetIndex = DatasetIndex.rename(columns={SampleIdColumn: "SampleId"})

    DatasetIndex = DatasetIndex.merge(ProductPresence, on="SampleId", how="left")

    for Column in ["HasTarget", "HasReference", "HasPlume", "HasCH4", "HasDem"]:
        if Column in DatasetIndex.columns:
            DatasetIndex[Column] = DatasetIndex[Column].fillna(False).astype(bool)

    if "ProductCount" in DatasetIndex.columns:
        DatasetIndex["ProductCount"] = DatasetIndex["ProductCount"].fillna(0).astype(int)

    if "Products" in DatasetIndex.columns:
        DatasetIndex["Products"] = DatasetIndex["Products"].fillna("")

    return DatasetIndex


def SummarizeColumns(DataFrame: pd.DataFrame) -> pd.DataFrame:
    """Resume columnas, tipos y nulos."""
    Rows = []

    for Column in DataFrame.columns:
        Rows.append(
            {
                "Column": Column,
                "Dtype": str(DataFrame[Column].dtype),
                "NonNull": int(DataFrame[Column].notna().sum()),
                "Null": int(DataFrame[Column].isna().sum()),
            }
        )

    return pd.DataFrame(Rows)
