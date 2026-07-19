#!/usr/bin/env python3
"""
Step06ApplyFeatureReadinessFilter.py

Aplica el filtro de feature readiness basado en la validez de MBMPPlus.

Contexto metodológico:
- MBMPPlus se calcula como corrección supervisada de fondo usando Ridge.
- El método requiere suficientes píxeles válidos de fondo.
- Las muestras inválidas se excluyen del dataset efectivo final.
- La exclusión se aplica a ConfigA, ConfigB y ConfigC para preservar comparabilidad.

Entradas:
- Tables/SplitTrain.csv
- Tables/SplitValidation.csv
- Tables/SplitTest.csv
- ConfigB/Tables/MBMPPlusValidityBySample.csv

Salidas:
- Tables/SplitTrainFeatureReady.csv
- Tables/SplitValidationFeatureReady.csv
- Tables/SplitTestFeatureReady.csv
- Tables/SplitFeatureReadySummary.csv
- Tables/ExcludedFeatureReadinessSamples.csv
- Audit/FeatureReadinessFilterAudit.json
- Logs/Step06ApplyFeatureReadinessFilter.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


SPLIT_FILES = {
    "Train": "SplitTrain.csv",
    "Validation": "SplitValidation.csv",
    "Test": "SplitTest.csv",
}


OUTPUT_SPLIT_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}


def LoadOriginalSplits(TablesDirectory: Path) -> dict[str, pd.DataFrame]:
    """Carga los splits originales."""
    Splits = {}

    for SplitName, FileName in SPLIT_FILES.items():
        PathItem = TablesDirectory / FileName

        if not PathItem.exists():
            raise FileNotFoundError(f"No existe el split original: {PathItem}")

        Table = pd.read_csv(PathItem)

        if "SampleId" not in Table.columns:
            raise KeyError(f"{PathItem} debe contener la columna SampleId.")

        if Table.empty:
            raise ValueError(f"El split original está vacío: {PathItem}")

        Table = Table.copy()
        Table["SampleId"] = Table["SampleId"].astype(str)
        Splits[SplitName] = Table

    return Splits


def LoadValidityTable(ValidityPath: Path) -> pd.DataFrame:
    """Carga la tabla de validez de MBMPPlus."""
    if not ValidityPath.exists():
        raise FileNotFoundError(
            f"No existe la tabla de validez MBMPPlus: {ValidityPath}. "
            "Ejecuta primero Step05DiagnoseMBMPPlusValidity.py."
        )

    Table = pd.read_csv(ValidityPath)

    RequiredColumns = [
        "Split",
        "SplitIndex",
        "SampleId",
        "Status",
        "FailureReason",
    ]

    Missing = [Column for Column in RequiredColumns if Column not in Table.columns]
    if Missing:
        raise KeyError(f"Faltan columnas en {ValidityPath}: {Missing}")

    Table = Table.copy()
    Table["SampleId"] = Table["SampleId"].astype(str)

    return Table


def BuildExcludedTable(ValidityTable: pd.DataFrame) -> pd.DataFrame:
    """Construye tabla de muestras excluidas."""
    Excluded = ValidityTable[ValidityTable["Status"] != "Valid"].copy()

    if Excluded.empty:
        return pd.DataFrame(
            columns=[
                "Split",
                "SplitIndex",
                "SampleId",
                "Status",
                "FailureReason",
                "ExclusionReason",
            ]
        )

    Excluded["ExclusionReason"] = (
        "Invalid MBMPPlus supervised background correction readiness"
    )

    PreferredColumns = [
        "Split",
        "SplitIndex",
        "SampleId",
        "Status",
        "FailureReason",
        "PlumePixels",
        "BackgroundPixels",
        "ValidMbmpInverseBackgroundPixels",
        "ValidAllPredictorBackgroundPixels",
        "ValidStrictBackgroundPixels",
        "WorstPredictorBand",
        "WorstPredictorValidBackgroundPixels",
        "ExclusionReason",
    ]

    ExistingColumns = [Column for Column in PreferredColumns if Column in Excluded.columns]
    ExtraColumns = [Column for Column in Excluded.columns if Column not in ExistingColumns]

    return Excluded[ExistingColumns + ExtraColumns]


def ApplyReadinessFilter(
    OriginalSplits: dict[str, pd.DataFrame],
    ExcludedTable: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Aplica exclusión por SampleId a cada split."""
    ExcludedIdsBySplit = {
        SplitName: set(
            ExcludedTable.loc[
                ExcludedTable["Split"] == SplitName,
                "SampleId",
            ].astype(str)
        )
        for SplitName in SPLIT_FILES.keys()
    }

    ReadySplits = {}
    SummaryRows = []

    for SplitName, OriginalTable in OriginalSplits.items():
        ExcludedIds = ExcludedIdsBySplit.get(SplitName, set())

        ReadyTable = OriginalTable[
            ~OriginalTable["SampleId"].astype(str).isin(ExcludedIds)
        ].copy()

        ReadyTable["FeatureReady"] = True
        ReadyTable["FeatureReadinessFilter"] = "MBMPPlusValidity"

        OriginalCount = int(len(OriginalTable))
        ExcludedCount = int(OriginalCount - len(ReadyTable))
        ReadyCount = int(len(ReadyTable))

        SummaryRows.append(
            {
                "Split": SplitName,
                "OriginalSamples": OriginalCount,
                "ExcludedSamples": ExcludedCount,
                "FeatureReadySamples": ReadyCount,
                "ExcludedPercentage": float(100.0 * ExcludedCount / max(1, OriginalCount)),
                "Filter": "MBMPPlusValidity",
                "AppliedToConfigs": "ConfigA,ConfigB,ConfigC",
            }
        )

        ReadySplits[SplitName] = ReadyTable

    TotalOriginal = sum(len(Table) for Table in OriginalSplits.values())
    TotalReady = sum(len(Table) for Table in ReadySplits.values())
    TotalExcluded = TotalOriginal - TotalReady

    SummaryRows.append(
        {
            "Split": "All",
            "OriginalSamples": int(TotalOriginal),
            "ExcludedSamples": int(TotalExcluded),
            "FeatureReadySamples": int(TotalReady),
            "ExcludedPercentage": float(100.0 * TotalExcluded / max(1, TotalOriginal)),
            "Filter": "MBMPPlusValidity",
            "AppliedToConfigs": "ConfigA,ConfigB,ConfigC",
        }
    )

    Summary = pd.DataFrame(SummaryRows)

    return ReadySplits, Summary


def ValidateReadySplits(
    ReadySplits: dict[str, pd.DataFrame],
    ExcludedTable: pd.DataFrame,
) -> None:
    """Valida que ningún SampleId excluido permanezca en los splits ready."""
    ExcludedIds = set(ExcludedTable["SampleId"].astype(str).tolist())

    for SplitName, Table in ReadySplits.items():
        RemainingExcluded = sorted(
            set(Table["SampleId"].astype(str).tolist()).intersection(ExcludedIds)
        )

        if RemainingExcluded:
            raise AssertionError(
                f"El split {SplitName} todavía contiene muestras excluidas: "
                f"{RemainingExcluded[:10]}"
            )

        if Table.empty:
            raise ValueError(f"El split FeatureReady quedó vacío: {SplitName}")


def Main() -> None:
    Parser = argparse.ArgumentParser(
        description="Aplica filtro FeatureReady basado en validez MBMPPlus."
    )
    Parser = AddCommonArguments(Parser)
    Parser.add_argument(
        "--ValidityTable",
        default=None,
        help=(
            "Ruta opcional a MBMPPlusValidityBySample.csv. "
            "Por defecto usa ConfigB/Tables/MBMPPlusValidityBySample.csv."
        ),
    )
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    LogPath = Paths.LogsDirectory / "Step06ApplyFeatureReadinessFilter.log"
    Logger = CreateLogger("Step06ApplyFeatureReadinessFilter", LogPath)

    ValidityPath = (
        Path(Args.ValidityTable)
        if Args.ValidityTable is not None
        else Paths.RunDirectory / "ConfigB" / "Tables" / "MBMPPlusValidityBySample.csv"
    )

    if not ValidityPath.is_absolute():
        ValidityPath = Paths.ProjectRoot / ValidityPath

    Logger.info("Loading original splits from: %s", Paths.TablesDirectory)
    OriginalSplits = LoadOriginalSplits(Paths.TablesDirectory)

    Logger.info("Loading validity table: %s", ValidityPath)
    ValidityTable = LoadValidityTable(ValidityPath)

    ExcludedTable = BuildExcludedTable(ValidityTable)

    ReadySplits, Summary = ApplyReadinessFilter(
        OriginalSplits=OriginalSplits,
        ExcludedTable=ExcludedTable,
    )

    ValidateReadySplits(
        ReadySplits=ReadySplits,
        ExcludedTable=ExcludedTable,
    )

    OutputTables = {}

    for SplitName, ReadyTable in ReadySplits.items():
        OutputPath = Paths.TablesDirectory / OUTPUT_SPLIT_FILES[SplitName]
        ReadyTable.to_csv(OutputPath, index=False)
        OutputTables[f"Split{SplitName}FeatureReady"] = OutputPath
        Logger.info("%s FeatureReady samples: %d", SplitName, len(ReadyTable))

    SummaryPath = Paths.TablesDirectory / "SplitFeatureReadySummary.csv"
    ExcludedPath = Paths.TablesDirectory / "ExcludedFeatureReadinessSamples.csv"
    AuditPath = Paths.AuditDirectory / "FeatureReadinessFilterAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    Summary.to_csv(SummaryPath, index=False)
    ExcludedTable.to_csv(ExcludedPath, index=False)

    Details = {
        "Filter": "MBMPPlusValidity",
        "AppliedToConfigs": ["ConfigA", "ConfigB", "ConfigC"],
        "Policy": "Exclude invalid MBMPPlus samples for all comparative experiments.",
        "NoFallbackApplied": True,
        "OriginalSamples": int(Summary.loc[Summary["Split"] == "All", "OriginalSamples"].iloc[0]),
        "ExcludedSamples": int(Summary.loc[Summary["Split"] == "All", "ExcludedSamples"].iloc[0]),
        "FeatureReadySamples": int(
            Summary.loc[Summary["Split"] == "All", "FeatureReadySamples"].iloc[0]
        ),
        "ExpectedAfterCurrentDiagnosis": {
            "Train": int(Summary.loc[Summary["Split"] == "Train", "FeatureReadySamples"].iloc[0]),
            "Validation": int(
                Summary.loc[Summary["Split"] == "Validation", "FeatureReadySamples"].iloc[0]
            ),
            "Test": int(Summary.loc[Summary["Split"] == "Test", "FeatureReadySamples"].iloc[0]),
        },
        "MethodologicalNote": (
            "The exclusion is applied uniformly to ConfigA, ConfigB and ConfigC "
            "to preserve comparability across experiments."
        ),
    }

    Audit = BuildAuditRecord(
        ScriptName="Step06ApplyFeatureReadinessFilter.py",
        RunTag=Args.RunTag,
        Parameters={
            "ValidityTable": str(ValidityPath),
            "Filter": "Status != Valid",
        },
        Inputs={
            "SplitTrain": str(Paths.TablesDirectory / "SplitTrain.csv"),
            "SplitValidation": str(Paths.TablesDirectory / "SplitValidation.csv"),
            "SplitTest": str(Paths.TablesDirectory / "SplitTest.csv"),
            "MBMPPlusValidityBySample": str(ValidityPath),
        },
        Outputs={
            **{Key: str(Value) for Key, Value in OutputTables.items()},
            "SplitFeatureReadySummary": str(SummaryPath),
            "ExcludedFeatureReadinessSamples": str(ExcludedPath),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details=Details,
    )

    WriteJson(Audit, AuditPath)

    RegisterItems = [
        ("Table", SummaryPath, "Resumen de splits FeatureReady."),
        ("Table", ExcludedPath, "Muestras excluidas por feature readiness."),
        ("Audit", AuditPath, "Auditoría del filtro FeatureReady."),
    ]

    for SplitName, OutputPath in OutputTables.items():
        RegisterItems.append(
            ("Table", OutputPath, f"Split final {SplitName}.")
        )

    for OutputType, OutputPath, Description in RegisterItems:
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step06ApplyFeatureReadinessFilter",
            Config="Project",
            Model="None",
            OutputType=OutputType,
            RelativePath=str(OutputPath.relative_to(Paths.RunDirectory)),
            Created=OutputPath.exists(),
            Description=Description,
        )

    Logger.info("Feature readiness filter completed.")
    Logger.info("Summary:\n%s", Summary.to_string(index=False))

    print("\n=== FeatureReady summary ===")
    print(Summary.to_string(index=False))

    print("\nExcluded samples:", len(ExcludedTable))
    print("Saved:", SummaryPath)
    print("Saved:", ExcludedPath)
    print("Saved:", AuditPath)


if __name__ == "__main__":
    Main()
