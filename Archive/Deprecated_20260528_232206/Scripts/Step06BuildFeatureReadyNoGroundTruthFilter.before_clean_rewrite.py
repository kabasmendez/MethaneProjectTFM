#!/usr/bin/env python3
"""
Step06BuildFeatureReadyNoGroundTruthFilter.py

Construye splits FeatureReady sin excluir muestras.

Uso metodológico:
- MBMPPlus es no supervisado.
- Ninguna feature de entrada usa máscara de pluma ni ground truth.
- Por tanto, no corresponde excluir muestras por validez de fondo supervisado.

Entradas:
- Outputs/Experiments/<RunTag>/Tables/SplitTrain.csv
- Outputs/Experiments/<RunTag>/Tables/SplitValidation.csv
- Outputs/Experiments/<RunTag>/Tables/SplitTest.csv

Salidas principales esperadas por Step07:
- Outputs/Experiments/<RunTag>/Tables/SplitTrainFeatureReady.csv
- Outputs/Experiments/<RunTag>/Tables/SplitValidationFeatureReady.csv
- Outputs/Experiments/<RunTag>/Tables/SplitTestFeatureReady.csv

Alias adicionales:
- FeatureReadyTrain.csv
- FeatureReadyValidation.csv
- FeatureReadyTest.csv

Resumen:
- FeatureReadySummary.csv
- FeatureReadyExclusionLog.csv
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

INPUT_SPLIT_FILES = {
    "Train": "SplitTrain.csv",
    "Validation": "SplitValidation.csv",
    "Test": "SplitTest.csv",
}

STEP07_READY_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}

ALIAS_READY_FILES = {
    "Train": "FeatureReadyTrain.csv",
    "Validation": "FeatureReadyValidation.csv",
    "Test": "FeatureReadyTest.csv",
}


def LoadSplitTable(TablesDirectory: Path, Split: str) -> tuple[pd.DataFrame, Path]:
    InputPath = TablesDirectory / INPUT_SPLIT_FILES[Split]

    if not InputPath.exists():
        raise FileNotFoundError(
            f"No existe la tabla de split requerida: {InputPath}. "
            f"Ejecuta Step01BuildDatasetIndex.py antes de este script."
        )

    Table = pd.read_csv(InputPath)

    if "SampleId" not in Table.columns:
        raise KeyError(f"La tabla {InputPath} no contiene columna SampleId.")

    return Table, InputPath


def BuildReadyTable(Table: pd.DataFrame) -> pd.DataFrame:
    Ready = Table.copy()

    Ready["FeatureReady"] = True
    Ready["FeatureReadyFilter"] = "None"
    Ready["FeatureReadyReason"] = "MBMPPlusUnsupervised_NoGroundTruthFilter"
    Ready["ExcludedByFeatureReadiness"] = False

    return Ready


def Main() -> None:
    Parser = argparse.ArgumentParser(
        description="Construye FeatureReady splits sin filtro ground-truth."
    )
    Parser = AddCommonArguments(Parser)

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    TablesDirectory = Paths.TablesDirectory
    TablesDirectory.mkdir(parents=True, exist_ok=True)

    SummaryRows = []

    for Split in SPLITS:
        SplitTable, InputPath = LoadSplitTable(TablesDirectory, Split)
        ReadyTable = BuildReadyTable(SplitTable)

        Step07OutputPath = TablesDirectory / STEP07_READY_FILES[Split]
        AliasOutputPath = TablesDirectory / ALIAS_READY_FILES[Split]

        ReadyTable.to_csv(Step07OutputPath, index=False)
        ReadyTable.to_csv(AliasOutputPath, index=False)

        SummaryRows.append(
            {
                "Split": Split,
                "InputPath": str(InputPath),
                "Step07OutputPath": str(Step07OutputPath),
                "AliasOutputPath": str(AliasOutputPath),
                "TotalSamples": int(len(SplitTable)),
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
    print("\nCreated files:")
    for Split in SPLITS:
        print("-", TablesDirectory / STEP07_READY_FILES[Split])
        print("-", TablesDirectory / ALIAS_READY_FILES[Split])


if __name__ == "__main__":
    Main()
