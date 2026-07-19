#!/usr/bin/env python3
"""
Step06BuildFeatureReadyNoGroundTruthFilter.py

Construye tablas FeatureReady sin excluir muestras por MBMPPlus supervisado.

Uso:
- MBMPPlus ya no usa ground truth.
- Por tanto no se debe excluir ninguna muestra por ValidBackgroundPixels.
- Copia SplitTrain/SplitValidation/SplitTest a FeatureReadyTrain/Validation/Test.

Salidas:
- Tables/FeatureReadyTrain.csv
- Tables/FeatureReadyValidation.csv
- Tables/FeatureReadyTest.csv
- Tables/FeatureReadySummary.csv
- Tables/FeatureReadyExclusionLog.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]


def FindSplitTable(TablesDirectory: Path, Split: str) -> Path:
    Candidates = [
        TablesDirectory / f"Split{Split}.csv",
        TablesDirectory / f"{Split}.csv",
        TablesDirectory / f"{Split}Split.csv",
    ]

    for Candidate in Candidates:
        if Candidate.exists():
            return Candidate

    raise FileNotFoundError(
        f"No encontré tabla para split {Split}. Probé: {Candidates}"
    )


def Main() -> None:
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--RunTag", required=True)
    Args = Parser.parse_args()

    RunRoot = ProjectRoot / "Outputs" / "Experiments" / Args.RunTag
    TablesDirectory = RunRoot / "Tables"
    TablesDirectory.mkdir(parents=True, exist_ok=True)

    Rows = []
    Exclusions = []

    for Split in ["Train", "Validation", "Test"]:
        InputPath = FindSplitTable(TablesDirectory, Split)
        OutputPath = TablesDirectory / f"FeatureReady{Split}.csv"

        Table = pd.read_csv(InputPath)
        Table = Table.copy()
        Table["FeatureReady"] = True
        Table["FeatureReadyReason"] = "NoGroundTruthFilter_MBMPPlusUnsupervised"

        Table.to_csv(OutputPath, index=False)

        Rows.append(
            {
                "Split": Split,
                "InputPath": str(InputPath),
                "OutputPath": str(OutputPath),
                "TotalSamples": len(Table),
                "FeatureReadySamples": len(Table),
                "ExcludedSamples": 0,
                "ExclusionReason": "None",
                "SplitSource": "FeatureReadyNoGroundTruthFilter",
            }
        )

    Summary = pd.DataFrame(Rows)
    SummaryPath = TablesDirectory / "FeatureReadySummary.csv"
    Summary.to_csv(SummaryPath, index=False)

    ExclusionLog = pd.DataFrame(
        Exclusions,
        columns=["Split", "SampleId", "Reason"],
    )
    ExclusionLogPath = TablesDirectory / "FeatureReadyExclusionLog.csv"
    ExclusionLog.to_csv(ExclusionLogPath, index=False)

    print("\n=== FeatureReady no-ground-truth filter completed ===")
    print("RunTag:", Args.RunTag)
    print("Summary:", SummaryPath)
    print(Summary.to_string(index=False))


if __name__ == "__main__":
    Main()
