#!/usr/bin/env python3
"""
Step05DiagnoseMBMPPlusValidity.py

Diagnóstico de validez de MBMPPlus estricto sobre todo el dataset efectivo.

Objetivo:
- medir cuántas muestras pueden calcular MBMPPlus bajo la lógica actual;
- identificar muestras que no tienen suficientes píxeles válidos de fondo;
- cuantificar impacto por split antes de decidir si se filtra, se usa fallback o se cambia metodología.

Este script NO crea tensores y NO modifica features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml, ValidateProjectConfig
from Source.FeatureEngineering import S2_BAND_INDEX, ComputeMbmpInverse
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.ReadTacoSample import ReadFullTacoSample
from Source.RunUtils import AddCommonArguments, ResolveProjectPath, ValidateCommonArguments
from Source.TacoIndex import GetSampleTable, LoadTacoDataset


SPLIT_FILES = {
    "Train": "SplitTrain.csv",
    "Validation": "SplitValidation.csv",
    "Test": "SplitTest.csv",
}


def LoadSplitTables(TablesDirectory: Path) -> dict[str, pd.DataFrame]:
    """Carga las tablas de split."""
    SplitTables = {}

    for SplitName, FileName in SPLIT_FILES.items():
        PathItem = TablesDirectory / FileName

        if not PathItem.exists():
            raise FileNotFoundError(f"No existe {PathItem}")

        Table = pd.read_csv(PathItem)

        if "SampleId" not in Table.columns:
            raise KeyError(f"{PathItem} debe contener columna SampleId.")

        SplitTables[SplitName] = Table

    return SplitTables


def DiagnoseStrictMbmpPlusValidity(
    Target: np.ndarray,
    Reference: np.ndarray,
    Plume: np.ndarray,
    MinimumBackgroundPixels: int = 50,
    Epsilon: float = 1e-6,
) -> dict[str, Any]:
    """
    Diagnostica la validez bajo la lógica estricta actual:

    - MBMP inverse debe ser finito.
    - Todos los ratios auxiliares Bk_target / Bk_reference deben ser finitos.
    - Se usan todas las bandas excepto B11 y B12.
    - Se exige al menos MinimumBackgroundPixels píxeles válidos de fondo.
    """
    Target = np.asarray(Target, dtype=np.float32)
    Reference = np.asarray(Reference, dtype=np.float32)
    PlumeMask = np.asarray(Plume > 0)

    if Target.shape != Reference.shape:
        raise ValueError(f"Target y Reference tienen formas distintas: {Target.shape} vs {Reference.shape}")

    if Target.ndim != 3:
        raise ValueError(f"Target debe ser 3D Bands x H x W. Recibido: {Target.shape}")

    Bands, Height, Width = Target.shape

    if PlumeMask.shape != (Height, Width):
        raise ValueError(f"PlumeMask shape {PlumeMask.shape} no coincide con {(Height, Width)}")

    Background = (~PlumeMask).reshape(-1)
    PlumePixels = PlumeMask.reshape(-1)

    MbmpInverse = ComputeMbmpInverse(Target, Reference, Epsilon=Epsilon).reshape(-1)

    ExcludedBands = {
        S2_BAND_INDEX["B11"],
        S2_BAND_INDEX["B12"],
    }

    PredictorBandIndices = [Index for Index in range(Bands) if Index not in ExcludedBands]

    TargetFlat = Target.reshape(Bands, -1)
    ReferenceFlat = Reference.reshape(Bands, -1)

    SafeReference = np.where(np.abs(ReferenceFlat) > Epsilon, ReferenceFlat, np.nan)
    TemporalRatios = TargetFlat / SafeReference

    Predictors = TemporalRatios[PredictorBandIndices].T

    ValidMbmpInverse = np.isfinite(MbmpInverse)
    ValidPredictorsAll = np.isfinite(Predictors).all(axis=1)

    ValidAll = ValidMbmpInverse & ValidPredictorsAll
    ValidBackground = ValidAll & Background

    PerBandRows = []
    WorstBand = None
    WorstValidBackground = None

    for BandIndex in PredictorBandIndices:
        BandName = None
        for CandidateName, CandidateIndex in S2_BAND_INDEX.items():
            if CandidateIndex == BandIndex:
                BandName = CandidateName
                break

        Ratio = TemporalRatios[BandIndex]
        ValidRatio = np.isfinite(Ratio)
        ValidRatioBackground = ValidRatio & Background

        ValidBackgroundCount = int(ValidRatioBackground.sum())

        PerBandRows.append(
            {
                "BandName": BandName or f"Band{BandIndex}",
                "BandIndex": int(BandIndex),
                "ValidRatioPixels": int(ValidRatio.sum()),
                "ValidRatioBackgroundPixels": ValidBackgroundCount,
                "ValidRatioBackgroundFraction": float(
                    ValidBackgroundCount / max(1, int(Background.sum()))
                ),
                "ReferenceZeroOrNearZeroPixels": int((np.abs(ReferenceFlat[BandIndex]) <= Epsilon).sum()),
            }
        )

        if WorstValidBackground is None or ValidBackgroundCount < WorstValidBackground:
            WorstValidBackground = ValidBackgroundCount
            WorstBand = BandName or f"Band{BandIndex}"

    Status = "Valid" if int(ValidBackground.sum()) >= MinimumBackgroundPixels else "Invalid"

    FailureReason = ""

    if int(Background.sum()) < MinimumBackgroundPixels:
        FailureReason = "NotEnoughBackgroundPixels"
    elif int((ValidMbmpInverse & Background).sum()) < MinimumBackgroundPixels:
        FailureReason = "NotEnoughValidMbmpInverseBackgroundPixels"
    elif int(ValidBackground.sum()) < MinimumBackgroundPixels:
        FailureReason = "NotEnoughStrictValidBackgroundPixels"
    else:
        FailureReason = "None"

    return {
        "Status": Status,
        "FailureReason": FailureReason,
        "Height": int(Height),
        "Width": int(Width),
        "PlumePixels": int(PlumePixels.sum()),
        "BackgroundPixels": int(Background.sum()),
        "ValidMbmpInversePixels": int(ValidMbmpInverse.sum()),
        "ValidMbmpInverseBackgroundPixels": int((ValidMbmpInverse & Background).sum()),
        "ValidAllPredictorPixels": int(ValidPredictorsAll.sum()),
        "ValidAllPredictorBackgroundPixels": int((ValidPredictorsAll & Background).sum()),
        "ValidStrictPixels": int(ValidAll.sum()),
        "ValidStrictBackgroundPixels": int(ValidBackground.sum()),
        "MinimumBackgroundPixels": int(MinimumBackgroundPixels),
        "PredictorBandIndices": ",".join(str(Index) for Index in PredictorBandIndices),
        "WorstPredictorBand": WorstBand,
        "WorstPredictorValidBackgroundPixels": int(WorstValidBackground or 0),
        "PerBandDiagnostics": PerBandRows,
    }


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Diagnostica validez de MBMPPlus en todo el dataset.")
    Parser = AddCommonArguments(Parser)
    Parser.add_argument(
        "--MinimumBackgroundPixels",
        type=int,
        default=50,
        help="Mínimo de píxeles válidos de fondo requerido para Ridge.",
    )
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    LogPath = Paths.LogsDirectory / "Step05DiagnoseMBMPPlusValidity.log"
    Logger = CreateLogger("Step05DiagnoseMBMPPlusValidity", LogPath)

    ProjectConfigPath = ResolveProjectPath(Paths.ProjectRoot, Args.ProjectConfig)
    ProjectConfig = LoadYaml(ProjectConfigPath)
    ValidateProjectConfig(ProjectConfig)

    SplitTables = LoadSplitTables(Paths.TablesDirectory)

    Dataset, DatasetInfo = LoadTacoDataset(
        ProjectConfig["Dataset"]["DataRoot"],
        ProjectConfig["Dataset"]["DatasetName"],
    )
    SampleTable = GetSampleTable(Dataset)
    ExpectedShapes = ProjectConfig.get("ExpectedShapes", {})

    Rows = []
    PerBandRows = []

    TotalSamples = sum(len(Table) for Table in SplitTables.values())
    Processed = 0

    for SplitName, SplitTable in SplitTables.items():
        Logger.info("Diagnosing split %s with %d samples", SplitName, len(SplitTable))

        for SplitIndex, Row in SplitTable.iterrows():
            SampleId = str(Row["SampleId"])
            Processed += 1

            if Processed == 1 or Processed % 50 == 0:
                Logger.info("Progress: %d/%d | %s | %s", Processed, TotalSamples, SplitName, SampleId)

            try:
                SampleData = ReadFullTacoSample(
                    Dataset=Dataset,
                    SampleTable=SampleTable,
                    SampleId=SampleId,
                    ExpectedShapes=ExpectedShapes,
                )

                Diagnostic = DiagnoseStrictMbmpPlusValidity(
                    Target=SampleData["Target"],
                    Reference=SampleData["Reference"],
                    Plume=SampleData["Plume"],
                    MinimumBackgroundPixels=Args.MinimumBackgroundPixels,
                )

                RowOut = {
                    "Split": SplitName,
                    "SplitIndex": int(SplitIndex),
                    "SampleId": SampleId,
                    **{Key: Value for Key, Value in Diagnostic.items() if Key != "PerBandDiagnostics"},
                    "ReadError": "",
                }

                Rows.append(RowOut)

                for BandDiagnostic in Diagnostic["PerBandDiagnostics"]:
                    PerBandRows.append(
                        {
                            "Split": SplitName,
                            "SplitIndex": int(SplitIndex),
                            "SampleId": SampleId,
                            **BandDiagnostic,
                        }
                    )

            except Exception as Error:
                Rows.append(
                    {
                        "Split": SplitName,
                        "SplitIndex": int(SplitIndex),
                        "SampleId": SampleId,
                        "Status": "ReadOrDiagnosticError",
                        "FailureReason": repr(Error),
                        "Height": np.nan,
                        "Width": np.nan,
                        "PlumePixels": np.nan,
                        "BackgroundPixels": np.nan,
                        "ValidMbmpInversePixels": np.nan,
                        "ValidMbmpInverseBackgroundPixels": np.nan,
                        "ValidAllPredictorPixels": np.nan,
                        "ValidAllPredictorBackgroundPixels": np.nan,
                        "ValidStrictPixels": np.nan,
                        "ValidStrictBackgroundPixels": np.nan,
                        "MinimumBackgroundPixels": Args.MinimumBackgroundPixels,
                        "PredictorBandIndices": "",
                        "WorstPredictorBand": "",
                        "WorstPredictorValidBackgroundPixels": np.nan,
                        "ReadError": repr(Error),
                    }
                )

    OutputTableDirectory = Paths.RunDirectory / "ConfigB" / "Tables"
    OutputAuditDirectory = Paths.RunDirectory / "ConfigB" / "Audit"
    OutputTableDirectory.mkdir(parents=True, exist_ok=True)
    OutputAuditDirectory.mkdir(parents=True, exist_ok=True)

    BySamplePath = OutputTableDirectory / "MBMPPlusValidityBySample.csv"
    ByBandPath = OutputTableDirectory / "MBMPPlusValidityByBand.csv"
    SummaryPath = OutputTableDirectory / "MBMPPlusValiditySummary.csv"
    AuditPath = OutputAuditDirectory / "MBMPPlusValidityAudit.json"

    BySample = pd.DataFrame(Rows)
    ByBand = pd.DataFrame(PerBandRows)

    BySample.to_csv(BySamplePath, index=False)
    ByBand.to_csv(ByBandPath, index=False)

    SummaryRows = []

    for SplitName, Group in BySample.groupby("Split"):
        Total = len(Group)
        Valid = int((Group["Status"] == "Valid").sum())
        Invalid = int((Group["Status"] != "Valid").sum())

        SummaryRows.append(
            {
                "Split": SplitName,
                "TotalSamples": Total,
                "ValidSamples": Valid,
                "InvalidSamples": Invalid,
                "InvalidPercentage": float(100.0 * Invalid / max(1, Total)),
            }
        )

    Total = len(BySample)
    Valid = int((BySample["Status"] == "Valid").sum())
    Invalid = int((BySample["Status"] != "Valid").sum())

    SummaryRows.append(
        {
            "Split": "All",
            "TotalSamples": Total,
            "ValidSamples": Valid,
            "InvalidSamples": Invalid,
            "InvalidPercentage": float(100.0 * Invalid / max(1, Total)),
        }
    )

    Summary = pd.DataFrame(SummaryRows)
    Summary.to_csv(SummaryPath, index=False)

    Audit = BuildAuditRecord(
        ScriptName="Step05DiagnoseMBMPPlusValidity.py",
        RunTag=Args.RunTag,
        Parameters={
            "MinimumBackgroundPixels": Args.MinimumBackgroundPixels,
            "Method": "Strict current MBMPPlus validity check",
        },
        Inputs={
            "SplitTrain": str(Paths.TablesDirectory / "SplitTrain.csv"),
            "SplitValidation": str(Paths.TablesDirectory / "SplitValidation.csv"),
            "SplitTest": str(Paths.TablesDirectory / "SplitTest.csv"),
            "Dataset": DatasetInfo,
        },
        Outputs={
            "BySample": str(BySamplePath),
            "ByBand": str(ByBandPath),
            "Summary": str(SummaryPath),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "TotalSamples": int(Total),
            "ValidSamples": int(Valid),
            "InvalidSamples": int(Invalid),
            "InvalidPercentage": float(100.0 * Invalid / max(1, Total)),
            "ImportantNote": (
                "This diagnostic does not change the MBMPPlus method. "
                "It only measures how many samples satisfy the current strict validity condition."
            ),
        },
    )

    WriteJson(Audit, AuditPath)

    Logger.info("Diagnostic completed.")
    Logger.info("BySample: %s", BySamplePath)
    Logger.info("ByBand: %s", ByBandPath)
    Logger.info("Summary: %s", SummaryPath)
    Logger.info("Audit: %s", AuditPath)

    print("\n=== MBMPPlus validity summary ===")
    print(Summary.to_string(index=False))


if __name__ == "__main__":
    Main()
