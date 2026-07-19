#!/usr/bin/env python3
"""
Step07BuildFeatures.py

Construye tensores de features y máscaras para entrenamiento.

Versión limpia para el flujo MBMPPlus no supervisado.

Principios:
- ConfigB usa MBMPPlus no supervisado.
- Ninguna feature de entrada usa máscara de pluma ni ground truth.
- La máscara de pluma solo se usa como target Y.
- Si --UseFeatureReadySplits está activo, usa:
    SplitTrainFeatureReady.csv
    SplitValidationFeatureReady.csv
    SplitTestFeatureReady.csv
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml
from Source.FeatureEngineering import BuildFeatureDictionary
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.ReadTacoSample import ReadFullTacoSample
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


SPLITS = ["Train", "Validation", "Test"]

ORIGINAL_SPLIT_FILES = {
    "Train": "SplitTrain.csv",
    "Validation": "SplitValidation.csv",
    "Test": "SplitTest.csv",
}

FEATURE_READY_SPLIT_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}


def LoadSplitTable(TablesDirectory: Path, Split: str, UseFeatureReadySplits: bool) -> tuple[pd.DataFrame, Path]:
    FileMap = FEATURE_READY_SPLIT_FILES if UseFeatureReadySplits else ORIGINAL_SPLIT_FILES
    PathItem = TablesDirectory / FileMap[Split]

    if not PathItem.exists():
        raise FileNotFoundError(
            f"No existe la tabla requerida para {Split}: {PathItem}. "
            f"Si usas --UseFeatureReadySplits, ejecuta primero "
            f"Step06BuildFeatureReadyNoGroundTruthFilter.py."
        )

    Table = pd.read_csv(PathItem)

    if "SampleId" not in Table.columns:
        raise KeyError(f"La tabla {PathItem} no contiene columna SampleId.")

    return Table, PathItem


def ResolveFeatureConfigs(Args: argparse.Namespace) -> list[str]:
    if Args.FeatureConfigs:
        return list(Args.FeatureConfigs)

    if Args.Configs:
        return list(Args.Configs)

    if Args.FeatureConfig:
        return [Args.FeatureConfig]

    return ["ConfigA", "ConfigB"]


def LoadFeatureConfig(ConfigName: str) -> dict[str, Any]:
    ConfigPath = ProjectRoot / "Configs" / f"{ConfigName}.yaml"

    if not ConfigPath.exists():
        raise FileNotFoundError(ConfigPath)

    Config = LoadYaml(ConfigPath)

    if "Features" not in Config:
        raise KeyError(f"{ConfigPath} no contiene campo Features.")

    if "InputChannels" not in Config:
        Config["InputChannels"] = len(Config["Features"])

    if int(Config["InputChannels"]) != len(Config["Features"]):
        raise ValueError(
            f"{ConfigName}: InputChannels={Config['InputChannels']} "
            f"pero len(Features)={len(Config['Features'])}."
        )

    return Config


def ReadSampleFromRow(Row: pd.Series, SplitTable: pd.DataFrame) -> dict[str, Any]:
    """
    Lee una muestra usando ReadFullTacoSample.

    Se soportan dos interfaces posibles:
    1. ReadFullTacoSample(Row)
    2. ReadFullTacoSample(SplitTable, SampleId)

    Esto evita depender de una firma rígida si el lector cambió entre versiones.
    """
    SampleId = str(Row["SampleId"])

    Errors = []

    try:
        Sample = ReadFullTacoSample(Row)
        if isinstance(Sample, dict):
            return Sample
    except Exception as Error:
        Errors.append(f"ReadFullTacoSample(Row): {Error}")

    try:
        Sample = ReadFullTacoSample(SplitTable, SampleId)
        if isinstance(Sample, dict):
            return Sample
    except Exception as Error:
        Errors.append(f"ReadFullTacoSample(SplitTable, SampleId): {Error}")

    Signature = inspect.signature(ReadFullTacoSample)

    raise RuntimeError(
        f"No se pudo leer SampleId={SampleId} con ReadFullTacoSample. "
        f"Firma detectada: {Signature}. Errores: {' | '.join(Errors)}"
    )


def NormalizeSampleKeys(SampleData: dict[str, Any]) -> dict[str, Any]:
    """
    Normaliza nombres esperados por FeatureEngineering.

    ReadFullTacoSample debería devolver:
    - Target
    - Reference
    - Plume

    Si devuelve otra capitalización compatible, se normaliza.
    """
    KeyMap = {str(Key).lower(): Key for Key in SampleData.keys()}

    def get_required(*Candidates: str) -> Any:
        for Candidate in Candidates:
            Lower = Candidate.lower()
            if Lower in KeyMap:
                return SampleData[KeyMap[Lower]]
        raise KeyError(
            f"No encontré ninguna llave {Candidates} en SampleData. "
            f"Disponibles: {list(SampleData.keys())}"
        )

    return {
        "Target": get_required("Target", "target"),
        "Reference": get_required("Reference", "reference"),
        "Plume": get_required("Plume", "plume", "Mask", "mask"),
        "Raw": SampleData,
    }


def BuildFeatureStackForSample(
    SampleData: dict[str, Any],
    FeatureNames: list[str],
    FeatureConfig: str,
) -> np.ndarray:
    Normalized = NormalizeSampleKeys(SampleData)

    FeatureDictionary = BuildFeatureDictionary(
        Target=Normalized["Target"],
        Reference=Normalized["Reference"],
        FeatureConfig=FeatureConfig,
        ContextMetadata=SampleData.get("ContextMetadata"),
    )

    Missing = [FeatureName for FeatureName in FeatureNames if FeatureName not in FeatureDictionary]

    if Missing:
        Available = ", ".join(sorted(FeatureDictionary.keys()))
        raise KeyError(
            f"Features no disponibles para {FeatureConfig}: {Missing}. "
            f"Disponibles: {Available}"
        )

    Stack = np.stack(
        [FeatureDictionary[FeatureName].astype(np.float32) for FeatureName in FeatureNames],
        axis=0,
    )

    Stack = np.nan_to_num(
        Stack,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    return Stack


def BuildMaskForSample(SampleData: dict[str, Any]) -> np.ndarray:
    Normalized = NormalizeSampleKeys(SampleData)

    Mask = (np.asarray(Normalized["Plume"]) > 0).astype(np.uint8)

    if Mask.ndim != 2:
        raise ValueError(f"Plume mask debe ser 2D. Shape recibido: {Mask.shape}")

    return Mask[None, :, :].astype(np.uint8)


def CreateOutputArrays(
    OutputDirectory: Path,
    Split: str,
    NumSamples: int,
    NumChannels: int,
    Height: int,
    Width: int,
) -> tuple[np.memmap, np.memmap, Path, Path]:
    OutputDirectory.mkdir(parents=True, exist_ok=True)

    FeaturePath = OutputDirectory / f"{Split}Features.npy"
    MaskPath = OutputDirectory / f"{Split}Masks.npy"

    Features = np.lib.format.open_memmap(
        FeaturePath,
        mode="w+",
        dtype=np.float32,
        shape=(NumSamples, NumChannels, Height, Width),
    )

    Masks = np.lib.format.open_memmap(
        MaskPath,
        mode="w+",
        dtype=np.uint8,
        shape=(NumSamples, 1, Height, Width),
    )

    return Features, Masks, FeaturePath, MaskPath


def BuildConfigFeatures(
    Paths: Any,
    FeatureConfig: str,
    FeatureConfigDict: dict[str, Any],
    SplitTables: dict[str, pd.DataFrame],
    SplitPaths: dict[str, Path],
    UseFeatureReadySplits: bool,
    MaxSamplesPerSplit: int | None,
    ClipValue: float,
    Logger: Any,
) -> dict[str, Any]:
    FeatureNames = list(FeatureConfigDict["Features"])
    NumChannels = len(FeatureNames)

    ConfigRoot = Paths.RunDirectory / FeatureConfig
    FeatureDirectory = ConfigRoot / "Features"
    TablesDirectory = ConfigRoot / "Tables"
    AuditDirectory = ConfigRoot / "Audit"

    FeatureDirectory.mkdir(parents=True, exist_ok=True)
    TablesDirectory.mkdir(parents=True, exist_ok=True)
    AuditDirectory.mkdir(parents=True, exist_ok=True)

    FirstSplit = SPLITS[0]
    FirstTable = SplitTables[FirstSplit]
    FirstRow = FirstTable.iloc[0]
    FirstSample = ReadSampleFromRow(FirstRow, FirstTable)
    FirstMask = BuildMaskForSample(FirstSample)

    Height = int(FirstMask.shape[-2])
    Width = int(FirstMask.shape[-1])

    BuildRows = []
    OutputRecords = []

    for Split in SPLITS:
        Table = SplitTables[Split].copy()
        OriginalCount = len(Table)

        if MaxSamplesPerSplit is not None:
            Table = Table.head(int(MaxSamplesPerSplit)).copy()

        NumSamples = len(Table)

        Logger.info(
            f"Construyendo {FeatureConfig} {Split}: "
            f"{NumSamples}/{OriginalCount} muestras, {NumChannels} canales."
        )

        FeaturesArray, MasksArray, FeaturePath, MaskPath = CreateOutputArrays(
            OutputDirectory=FeatureDirectory,
            Split=Split,
            NumSamples=NumSamples,
            NumChannels=NumChannels,
            Height=Height,
            Width=Width,
        )

        for Index, (_, Row) in enumerate(Table.iterrows()):
            SampleData = ReadSampleFromRow(Row, SplitTables[Split])

            FeatureStack = BuildFeatureStackForSample(
                SampleData=SampleData,
                FeatureNames=FeatureNames,
                FeatureConfig=FeatureConfig,
            )

            Mask = BuildMaskForSample(SampleData)

            if FeatureStack.shape != (NumChannels, Height, Width):
                raise ValueError(
                    f"{FeatureConfig} {Split} index {Index}: "
                    f"FeatureStack shape {FeatureStack.shape}, esperado "
                    f"{(NumChannels, Height, Width)}"
                )

            if Mask.shape != (1, Height, Width):
                raise ValueError(
                    f"{FeatureConfig} {Split} index {Index}: "
                    f"Mask shape {Mask.shape}, esperado {(1, Height, Width)}"
                )

            if ClipValue is not None:
                FeatureStack = np.clip(FeatureStack, -float(ClipValue), float(ClipValue))

            FeatureStack = np.nan_to_num(
                FeatureStack,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

            FeaturesArray[Index] = FeatureStack
            MasksArray[Index] = Mask.astype(np.uint8)

            if Index == 0 or (Index + 1) % 50 == 0 or (Index + 1) == NumSamples:
                Logger.info(f"Writing {FeatureConfig} {Split}: {Index + 1}/{NumSamples}")

        FeaturesArray.flush()
        MasksArray.flush()

        BuildRows.append(
            {
                "RunTag": Paths.RunTag,
                "FeatureConfig": FeatureConfig,
                "Split": Split,
                "SplitSource": "FeatureReady" if UseFeatureReadySplits else "Original",
                "InputSplitPath": str(SplitPaths[Split]),
                "Samples": int(NumSamples),
                "OriginalSplitSamples": int(OriginalCount),
                "Channels": int(NumChannels),
                "Height": int(Height),
                "Width": int(Width),
                "FeaturePath": str(FeaturePath),
                "MaskPath": str(MaskPath),
                "Features": ", ".join(FeatureNames),
                "ClipValue": float(ClipValue) if ClipValue is not None else np.nan,
            }
        )

        OutputRecords.append(
            {
                "Split": Split,
                "FeaturePath": str(FeaturePath),
                "MaskPath": str(MaskPath),
            }
        )

    Summary = pd.DataFrame(BuildRows)
    SummaryPath = TablesDirectory / "FeatureBuildSummary.csv"
    Summary.to_csv(SummaryPath, index=False)

    FeatureListPath = TablesDirectory / "FeatureList.csv"
    pd.DataFrame(
        {
            "FeatureIndex": list(range(len(FeatureNames))),
            "FeatureName": FeatureNames,
        }
    ).to_csv(FeatureListPath, index=False)

    AuditPath = AuditDirectory / "FeatureBuildAudit.json"

    Audit = BuildAuditRecord(
        ScriptName="Step07BuildFeatures.py",
        RunTag=Paths.RunTag,
        Parameters={
            "FeatureConfig": FeatureConfig,
            "Features": FeatureNames,
            "InputChannels": NumChannels,
            "UseFeatureReadySplits": bool(UseFeatureReadySplits),
            "MaxSamplesPerSplit": MaxSamplesPerSplit,
            "ClipValue": ClipValue,
        },
        Inputs={Split: str(SplitPaths[Split]) for Split in SPLITS},
        Outputs={
            "FeatureDirectory": str(FeatureDirectory),
            "FeatureBuildSummary": str(SummaryPath),
            "FeatureList": str(FeatureListPath),
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "Methodology": {
                "FeatureInputsUseGroundTruth": False,
                "MaskUsage": "Segmentation target only.",
                "MBMPPlus": {
                    "Variant": "UnsupervisedRidgeCleaning",
                    "UsesGroundTruthMask": False,
                    "UsesPlumeMask": False,
                    "FormulaBase": "MBMP = (TargetB12 - ReferenceB12) / ReferenceB12",
                },
            },
            "Outputs": OutputRecords,
        },
    )

    WriteJson(Audit, AuditPath)

    Logger.info(f"{FeatureConfig} construido correctamente. Summary: {SummaryPath}")

    return {
        "FeatureConfig": FeatureConfig,
        "SummaryPath": str(SummaryPath),
        "AuditPath": str(AuditPath),
    }


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Construye tensores de features.")
    Parser = AddCommonArguments(Parser)

    Parser.add_argument("--ProjectConfig", default="ProjectConfig")
    Parser.add_argument("--FeatureConfig", default=None, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--FeatureConfigs", nargs="+", default=None, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--Configs", nargs="+", default=None, choices=["ConfigA", "ConfigB", "ConfigC"])
    Parser.add_argument("--UseFeatureReadySplits", action="store_true")
    Parser.add_argument("--MaxSamplesPerSplit", type=int, default=None)
    Parser.add_argument("--ClipValue", type=float, default=8.0)

    Args = Parser.parse_args()
    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)

    LogPath = Paths.LogsDirectory / "Step07BuildFeatures.log"
    Logger = CreateLogger("Step07BuildFeatures", LogPath)

    FeatureConfigs = ResolveFeatureConfigs(Args)

    Logger.info(f"RunTag: {Args.RunTag}")
    Logger.info(f"FeatureConfigs: {FeatureConfigs}")
    Logger.info(f"UseFeatureReadySplits: {Args.UseFeatureReadySplits}")

    SplitTables = {}
    SplitPaths = {}

    for Split in SPLITS:
        Table, PathItem = LoadSplitTable(
            TablesDirectory=Paths.TablesDirectory,
            Split=Split,
            UseFeatureReadySplits=bool(Args.UseFeatureReadySplits),
        )

        SplitTables[Split] = Table
        SplitPaths[Split] = PathItem

        Logger.info(f"{Split}: {len(Table)} muestras desde {PathItem}")

    Results = []

    for FeatureConfig in FeatureConfigs:
        FeatureConfigDict = LoadFeatureConfig(FeatureConfig)

        Result = BuildConfigFeatures(
            Paths=Paths,
            FeatureConfig=FeatureConfig,
            FeatureConfigDict=FeatureConfigDict,
            SplitTables=SplitTables,
            SplitPaths=SplitPaths,
            UseFeatureReadySplits=bool(Args.UseFeatureReadySplits),
            MaxSamplesPerSplit=Args.MaxSamplesPerSplit,
            ClipValue=Args.ClipValue,
            Logger=Logger,
        )

        Results.append(Result)

    GlobalSummaryPath = Paths.TablesDirectory / "FeatureBuildGlobalSummary.csv"
    pd.DataFrame(Results).to_csv(GlobalSummaryPath, index=False)

    Logger.info("Step07BuildFeatures completado.")

    print("\n=== Step07BuildFeatures completed ===")
    print("RunTag:", Args.RunTag)
    print("FeatureConfigs:", FeatureConfigs)
    print("Global summary:", GlobalSummaryPath)


if __name__ == "__main__":
    Main()
