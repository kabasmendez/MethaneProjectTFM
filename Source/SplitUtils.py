"""
SplitUtils.py

Creación y validación de splits Train/Validation/Test.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def CreateTrainValidationTestSplit(
    FilteredTable: pd.DataFrame,
    ProjectConfig: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea split reproducible 70/15/15 o según ProjectConfig."""
    if "SampleId" not in FilteredTable.columns:
        raise KeyError("FilteredTable debe contener columna SampleId.")

    Seed = int(ProjectConfig["Seed"])
    SplitConfig = ProjectConfig["Split"]

    TrainRatio = float(SplitConfig["Train"])
    ValidationRatio = float(SplitConfig["Validation"])
    TestRatio = float(SplitConfig["Test"])

    Total = TrainRatio + ValidationRatio + TestRatio
    if abs(Total - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones de split deben sumar 1.0. Suma: {Total}")

    if len(FilteredTable) < 10:
        raise ValueError("No hay suficientes muestras para crear splits.")

    TrainTable, TempTable = train_test_split(
        FilteredTable,
        train_size=TrainRatio,
        shuffle=True,
        random_state=Seed,
    )

    RelativeValidationRatio = ValidationRatio / (ValidationRatio + TestRatio)

    ValidationTable, TestTable = train_test_split(
        TempTable,
        train_size=RelativeValidationRatio,
        shuffle=True,
        random_state=Seed,
    )

    TrainTable = TrainTable.copy()
    ValidationTable = ValidationTable.copy()
    TestTable = TestTable.copy()

    TrainTable["Split"] = "Train"
    ValidationTable["Split"] = "Validation"
    TestTable["Split"] = "Test"

    SplitAll = pd.concat([TrainTable, ValidationTable, TestTable], ignore_index=True)

    SplitSummary = pd.DataFrame(
        [
            {
                "Split": "Train",
                "Count": int(len(TrainTable)),
                "Percentage": round(100 * len(TrainTable) / len(SplitAll), 3),
            },
            {
                "Split": "Validation",
                "Count": int(len(ValidationTable)),
                "Percentage": round(100 * len(ValidationTable) / len(SplitAll), 3),
            },
            {
                "Split": "Test",
                "Count": int(len(TestTable)),
                "Percentage": round(100 * len(TestTable) / len(SplitAll), 3),
            },
        ]
    )

    ValidateNoSplitOverlap(SplitAll)

    return SplitAll.reset_index(drop=True), SplitSummary


def ValidateNoSplitOverlap(SplitAll: pd.DataFrame) -> None:
    """Verifica que los splits no tengan SampleId en común."""
    if "SampleId" not in SplitAll.columns:
        raise KeyError("SplitAll debe contener columna SampleId.")

    RequiredSplits = ["Train", "Validation", "Test"]
    SplitSets = {}

    for SplitName in RequiredSplits:
        SplitSets[SplitName] = set(SplitAll.loc[SplitAll["Split"] == SplitName, "SampleId"])

    OverlapTrainValidation = SplitSets["Train"].intersection(SplitSets["Validation"])
    OverlapTrainTest = SplitSets["Train"].intersection(SplitSets["Test"])
    OverlapValidationTest = SplitSets["Validation"].intersection(SplitSets["Test"])

    Problems = {
        "TrainValidation": len(OverlapTrainValidation),
        "TrainTest": len(OverlapTrainTest),
        "ValidationTest": len(OverlapValidationTest),
    }

    Problems = {Key: Value for Key, Value in Problems.items() if Value > 0}

    if Problems:
        raise ValueError(f"Hay solapamiento entre splits: {Problems}")
