#!/usr/bin/env python3
"""
Step06BuildFeatureReadyNoGroundTruthFilter.py

Crea splits FeatureReady sin excluir muestras.

Decisión metodológica:
- ConfigB usa MBMPPlus no supervisado.
- MBMPPlus no usa Plume ni ground truth.
- No se requiere excluir muestras por validez de fondo supervisado.

Entradas:
- Tables/SplitTrain.csv
- Tables/SplitValidation.csv
- Tables/SplitTest.csv

Salidas usadas por Step07:
- Tables/SplitTrainFeatureReady.csv
- Tables/SplitValidationFeatureReady.csv
- Tables/SplitTestFeatureReady.csv

Alias informativos:
- Tables/FeatureReadyTrain.csv
- Tables/FeatureReadyValidation.csv
- Tables/FeatureReadyTest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


SPLITS = ["Train", "Validation", "Test"]

INPUT_FILES = {
    "Train": "SplitTrain.csv",
    "Validation": "SplitValidation.csv",
    "Test": "SplitTest.csv",
}

STEP07_OUTPUT_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}

ALIAS_OUTPUT_FILES = {
    "Train": "FeatureReadyTrain.csv",
    "Validation": "FeatureReadyValidation.csv",
    "Test": "FeatureReadyTest.csv",
}


def BuildReadyTable(InputTable: pd.DataFrame) -> pd.DataFrame:
    Table = InputTable.copy()

    if "SampleId" not in Table.columns:
        raise KeyError("La tabla de split no contiene columna SampleId.")

    Table["FeatureReady"] = True
    Table["FeatureReadyFilter"] = "None"
    Table["FeatureReadyReason"] = "MBMPPlusUnsupervised_NoGroundTruthFilter"
    Table["ExcludedByFeatureReadiness"] = False

    return Table


def Main() -> None:
    Parser = argparse.ArgumentParser(
        description="Crea FeatureReady splits sin filtro basado en ground truth."
    )
    Parser = AddCommonArguments(Parser)

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    TablesDirectory = Paths.TablesDirectory
    TablesDirectory.mkdir(parents=True, exist_ok=True)

    SummaryRows = []

    for Split in SPLITS:
        InputPath = TablesDirectory / INPUT_FILES[Split]

        if not InputPath.exists():
            raise FileNotFoundError(
                f"No existe {InputPath}. Ejecuta Step01BuildDatasetIndex.py primero."
            )

        InputTable = pd.read_csv(InputPath)
        ReadyTable = BuildReadyTable(InputTable)

        Step07OutputPath = TablesDirectory / STEP07_OUTPUT_FILES[Split]
        AliasOutputPath = TablesDirectory / ALIAS_OUTPUT_FILES[Split]

        ReadyTable.to_csv(Step07OutputPath, index=False)
        ReadyTable.to_csv(AliasOutputPath, index=False)

        SummaryRows.append(
            {
                "Split": Split,
                "InputPath": str(InputPath),
                "Step07OutputPath": str(Step07OutputPath),
                "AliasOutputPath": str(AliasOutputPath),
                "TotalSamples": int(len(InputTable)),
                "FeatureReadySamples": int(len(ReadyTable)),
                "ExcludedSamples": 0,
                "FeatureReadyFilter": "None",
                "FeatureReadyReason": "MBMPPlusUnsupervised_NoGroundTruthFilter",
                "SplitSource": "OriginalSplit_NoGroundTruthFilter",
            }
        )

    Summary = pd.DataFrame(SummaryRows)
    SummaryPath = TablesDirectory / "FeatureReadySummary.csv"
    Summary.to_csv(SummaryPath, index=False)

    ExclusionLog = pd.DataFrame(
        columns=[
            "Split",
            "SampleId",
            "Excluded",
            "ExclusionReason",
            "FeatureReadyFilter",
        ]
    )
    ExclusionLogPath = TablesDirectory / "FeatureReadyExclusionLog.csv"
    ExclusionLog.to_csv(ExclusionLogPath, index=False)

    print("\n=== FeatureReady no-ground-truth filter completed ===")
    print("RunTag:", Args.RunTag)
    print("Summary:", SummaryPath)
    print(Summary.to_string(index=False))


if __name__ == "__main__":
    Main()
