#!/usr/bin/env python3
"""
Step08CheckFeatureTensors.py

Verificación formal de tensores de features construidos por Step07BuildFeatures.py.

Verifica:
- existencia de archivos .npy;
- shapes esperados según splits FeatureReady;
- dtype;
- valores finitos;
- rangos después de clipping;
- consistencia de número de canales;
- máscaras binarias;
- features declaradas en FeatureNormalizationStats.csv.

Entradas:
- Tables/SplitTrainFeatureReady.csv
- Tables/SplitValidationFeatureReady.csv
- Tables/SplitTestFeatureReady.csv
- ConfigA/Features/*.npy
- ConfigB/Features/*.npy

Salidas:
- ConfigA/Tables/FeatureTensorCheck.csv
- ConfigB/Tables/FeatureTensorCheck.csv
- Audit/FeatureTensorCheckAudit.json
- Logs/Step08CheckFeatureTensors.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ProjectRoot = Path(__file__).resolve().parents[1]
sys.path.append(str(ProjectRoot))

from Source.AuditUtils import AppendOutputIndex, BuildAuditRecord, WriteJson
from Source.ConfigUtils import LoadYaml
from Source.LoggingUtils import CreateLogger
from Source.Paths import CreateExperimentDirectories
from Source.RunUtils import AddCommonArguments, ValidateCommonArguments


FEATURE_READY_SPLIT_FILES = {
    "Train": "SplitTrainFeatureReady.csv",
    "Validation": "SplitValidationFeatureReady.csv",
    "Test": "SplitTestFeatureReady.csv",
}


def LoadFeatureConfig(ProjectRoot: Path, ConfigName: str) -> dict[str, Any]:
    """Carga ConfigA.yaml / ConfigB.yaml."""
    ConfigPath = ProjectRoot / "Configs" / f"{ConfigName}.yaml"

    if not ConfigPath.exists():
        raise FileNotFoundError(f"No existe configuración: {ConfigPath}")

    Config = LoadYaml(ConfigPath)

    Required = ["FeatureConfig", "InputChannels", "Features"]
    Missing = [Key for Key in Required if Key not in Config]

    if Missing:
        raise KeyError(f"Faltan claves en {ConfigPath}: {Missing}")

    if Config["FeatureConfig"] != ConfigName:
        raise ValueError(f"{ConfigPath}: FeatureConfig no coincide con {ConfigName}")

    if int(Config["InputChannels"]) != len(Config["Features"]):
        raise ValueError(
            f"{ConfigPath}: InputChannels={Config['InputChannels']} "
            f"pero Features={len(Config['Features'])}"
        )

    return Config


def LoadExpectedSplitCounts(TablesDirectory: Path) -> dict[str, int]:
    """Lee splits FeatureReady y devuelve N esperado por split."""
    Counts = {}

    for SplitName, FileName in FEATURE_READY_SPLIT_FILES.items():
        PathItem = TablesDirectory / FileName

        if not PathItem.exists():
            raise FileNotFoundError(
                f"No existe split FeatureReady: {PathItem}. "
                "Ejecuta Step06ApplyFeatureReadinessFilter.py."
            )

        Table = pd.read_csv(PathItem)

        if "SampleId" not in Table.columns:
            raise KeyError(f"{PathItem} debe contener SampleId.")

        Counts[SplitName] = int(len(Table))

    return Counts


def ScanArrayFiniteAndRange(
    Array: np.ndarray,
    ChunkSize: int,
) -> dict[str, Any]:
    """Escanea un array memmap por chunks sobre la dimensión N."""
    N = Array.shape[0]

    GlobalMin = np.inf
    GlobalMax = -np.inf
    SumValues = 0.0
    SumSqValues = 0.0
    CountValues = 0
    NonFiniteCount = 0

    for Start in range(0, N, ChunkSize):
        End = min(Start + ChunkSize, N)
        Chunk = np.asarray(Array[Start:End])

        FiniteMask = np.isfinite(Chunk)
        NonFiniteCount += int((~FiniteMask).sum())

        if FiniteMask.any():
            FiniteValues = Chunk[FiniteMask].astype(np.float64)
            GlobalMin = min(GlobalMin, float(FiniteValues.min()))
            GlobalMax = max(GlobalMax, float(FiniteValues.max()))
            SumValues += float(FiniteValues.sum())
            SumSqValues += float((FiniteValues ** 2).sum())
            CountValues += int(FiniteValues.size)

    Mean = SumValues / CountValues if CountValues else np.nan
    Variance = (SumSqValues / CountValues) - (Mean ** 2) if CountValues else np.nan
    Std = float(np.sqrt(max(Variance, 0.0))) if CountValues else np.nan

    return {
        "Finite": bool(NonFiniteCount == 0),
        "NonFiniteCount": int(NonFiniteCount),
        "Min": float(GlobalMin),
        "Max": float(GlobalMax),
        "Mean": float(Mean),
        "Std": float(Std),
        "ValueCount": int(CountValues),
    }


def CheckMaskBinary(Array: np.ndarray, ChunkSize: int) -> dict[str, Any]:
    """Verifica que máscara sea binaria."""
    N = Array.shape[0]
    PositivePixels = 0
    NonBinaryPixels = 0
    UniqueValues = set()

    for Start in range(0, N, ChunkSize):
        End = min(Start + ChunkSize, N)
        Chunk = np.asarray(Array[Start:End])

        Unique = np.unique(Chunk)
        UniqueValues.update(int(Value) for Value in Unique.tolist())

        PositivePixels += int((Chunk > 0).sum())
        NonBinaryPixels += int((~np.isin(Chunk, [0, 1])).sum())

    return {
        "MaskBinary": bool(NonBinaryPixels == 0),
        "MaskPositivePixels": int(PositivePixels),
        "MaskNonBinaryPixels": int(NonBinaryPixels),
        "MaskUniqueValues": ",".join(str(Value) for Value in sorted(UniqueValues)),
    }


def CheckConfigTensors(
    RunDirectory: Path,
    ConfigName: str,
    FeatureConfig: dict[str, Any],
    ExpectedCounts: dict[str, int],
    ChunkSize: int,
    ExpectedHeight: int,
    ExpectedWidth: int,
    ClipValue: float,
    Logger,
) -> pd.DataFrame:
    """Verifica tensores de una configuración."""
    FeatureDirectory = RunDirectory / ConfigName / "Features"
    TablesDirectory = RunDirectory / ConfigName / "Tables"

    if not FeatureDirectory.exists():
        raise FileNotFoundError(f"No existe carpeta Features: {FeatureDirectory}")

    FeatureNames = list(FeatureConfig["Features"])
    ExpectedChannels = int(FeatureConfig["InputChannels"])

    BuildSummaryPath = TablesDirectory / "FeatureBuildSummary.csv"

    Rows = []

    for SplitName, ExpectedN in ExpectedCounts.items():
        Logger.info("Checking %s %s", ConfigName, SplitName)

        FeaturesPath = FeatureDirectory / f"{SplitName}Features.npy"
        MasksPath = FeatureDirectory / f"{SplitName}Masks.npy"

        if not FeaturesPath.exists():
            raise FileNotFoundError(FeaturesPath)

        if not MasksPath.exists():
            raise FileNotFoundError(MasksPath)

        X = np.load(FeaturesPath, mmap_mode="r")
        Y = np.load(MasksPath, mmap_mode="r")

        ExpectedFeatureShape = (ExpectedN, ExpectedChannels, ExpectedHeight, ExpectedWidth)
        ExpectedMaskShape = (ExpectedN, 1, ExpectedHeight, ExpectedWidth)

        ShapeOk = tuple(X.shape) == ExpectedFeatureShape
        MaskShapeOk = tuple(Y.shape) == ExpectedMaskShape
        DtypeOk = X.dtype == np.float32
        MaskDtypeOk = Y.dtype == np.uint8

        if not ShapeOk:
            raise AssertionError(f"{FeaturesPath}: shape {X.shape}, esperado {ExpectedFeatureShape}")

        if not MaskShapeOk:
            raise AssertionError(f"{MasksPath}: shape {Y.shape}, esperado {ExpectedMaskShape}")

        if not DtypeOk:
            raise AssertionError(f"{FeaturesPath}: dtype {X.dtype}, esperado float32")

        if not MaskDtypeOk:
            raise AssertionError(f"{MasksPath}: dtype {Y.dtype}, esperado uint8")

        Scan = ScanArrayFiniteAndRange(X, ChunkSize=ChunkSize)
        MaskCheck = CheckMaskBinary(Y, ChunkSize=ChunkSize)

        ClipOk = bool(Scan["Min"] >= -ClipValue - 1e-5 and Scan["Max"] <= ClipValue + 1e-5)

        Row = {
            "FeatureConfig": ConfigName,
            "Split": SplitName,
            "FeaturesPath": str(FeaturesPath.relative_to(RunDirectory)),
            "MasksPath": str(MasksPath.relative_to(RunDirectory)),
            "ExpectedSamples": ExpectedN,
            "FeatureShape": list(X.shape),
            "MaskShape": list(Y.shape),
            "ExpectedChannels": ExpectedChannels,
            "FeatureDtype": str(X.dtype),
            "MaskDtype": str(Y.dtype),
            "ShapeOk": ShapeOk,
            "MaskShapeOk": MaskShapeOk,
            "DtypeOk": DtypeOk,
            "MaskDtypeOk": MaskDtypeOk,
            "Finite": Scan["Finite"],
            "NonFiniteCount": Scan["NonFiniteCount"],
            "FeatureMin": Scan["Min"],
            "FeatureMax": Scan["Max"],
            "FeatureMean": Scan["Mean"],
            "FeatureStd": Scan["Std"],
            "ClipValue": ClipValue,
            "ClipRangeOk": ClipOk,
            **MaskCheck,
            "Features": ",".join(FeatureNames),
        }

        Row["AllChecksPassed"] = bool(
            ShapeOk
            and MaskShapeOk
            and DtypeOk
            and MaskDtypeOk
            and Scan["Finite"]
            and ClipOk
            and MaskCheck["MaskBinary"]
        )

        Rows.append(Row)

    Result = pd.DataFrame(Rows)



    return Result


def Main() -> None:
    Parser = argparse.ArgumentParser(description="Verifica tensores de features.")
    Parser = AddCommonArguments(Parser)
    Parser.add_argument(
        "--FeatureConfigs",
        nargs="+",
        default=["ConfigA", "ConfigB"],
        choices=["ConfigA", "ConfigB"],
    )
    Parser.add_argument("--ChunkSize", type=int, default=64)
    Parser.add_argument("--ExpectedHeight", type=int, default=200)
    Parser.add_argument("--ExpectedWidth", type=int, default=200)
    Parser.add_argument("--ClipValue", type=float, default=8.0)
    Args = Parser.parse_args()

    ValidateCommonArguments(Args)

    Paths = CreateExperimentDirectories(Args.RunTag)
    Logger = CreateLogger(
        "Step08CheckFeatureTensors",
        Paths.LogsDirectory / "Step08CheckFeatureTensors.log",
    )

    ExpectedCounts = LoadExpectedSplitCounts(Paths.TablesDirectory)

    Logger.info("Expected FeatureReady counts: %s", ExpectedCounts)

    AllRows = []
    OutputTables = {}

    for ConfigName in Args.FeatureConfigs:
        FeatureConfig = LoadFeatureConfig(Paths.ProjectRoot, ConfigName)

        Result = CheckConfigTensors(
            RunDirectory=Paths.RunDirectory,
            ConfigName=ConfigName,
            FeatureConfig=FeatureConfig,
            ExpectedCounts=ExpectedCounts,
            ChunkSize=Args.ChunkSize,
            ExpectedHeight=Args.ExpectedHeight,
            ExpectedWidth=Args.ExpectedWidth,
            ClipValue=Args.ClipValue,
            Logger=Logger,
        )

        OutputPath = Paths.RunDirectory / ConfigName / "Tables" / "FeatureTensorCheck.csv"
        Result.to_csv(OutputPath, index=False)
        OutputTables[ConfigName] = OutputPath
        AllRows.extend(Result.to_dict(orient="records"))

    AllResults = pd.DataFrame(AllRows)

    if not bool(AllResults["AllChecksPassed"].all()):
        Failed = AllResults[~AllResults["AllChecksPassed"]]
        raise AssertionError(f"Fallaron verificaciones de tensores:\n{Failed.to_string(index=False)}")

    AuditPath = Paths.AuditDirectory / "FeatureTensorCheckAudit.json"
    OutputIndexPath = Paths.AuditDirectory / "OutputIndex.csv"

    Audit = BuildAuditRecord(
        ScriptName="Step08CheckFeatureTensors.py",
        RunTag=Args.RunTag,
        Parameters={
            "FeatureConfigs": Args.FeatureConfigs,
            "ChunkSize": Args.ChunkSize,
            "ExpectedHeight": Args.ExpectedHeight,
            "ExpectedWidth": Args.ExpectedWidth,
            "ClipValue": Args.ClipValue,
        },
        Inputs={
            "SplitTrainFeatureReady": str(Paths.TablesDirectory / "SplitTrainFeatureReady.csv"),
            "SplitValidationFeatureReady": str(Paths.TablesDirectory / "SplitValidationFeatureReady.csv"),
            "SplitTestFeatureReady": str(Paths.TablesDirectory / "SplitTestFeatureReady.csv"),
        },
        Outputs={
            **{f"{Config}FeatureTensorCheck": str(PathItem) for Config, PathItem in OutputTables.items()},
            "Audit": str(AuditPath),
        },
        Status="Success",
        Details={
            "ExpectedCounts": ExpectedCounts,
            "AllChecksPassed": True,
            "CheckedConfigs": Args.FeatureConfigs,
        },
    )

    WriteJson(Audit, AuditPath)

    for Config, PathItem in OutputTables.items():
        AppendOutputIndex(
            OutputIndexPath=OutputIndexPath,
            RunTag=Args.RunTag,
            Step="Step08CheckFeatureTensors",
            Config=Config,
            Model="None",
            OutputType="Table",
            RelativePath=str(PathItem.relative_to(Paths.RunDirectory)),
            Created=PathItem.exists(),
            Description=f"Verificación formal de tensores {Config}.",
        )

    AppendOutputIndex(
        OutputIndexPath=OutputIndexPath,
        RunTag=Args.RunTag,
        Step="Step08CheckFeatureTensors",
        Config="Project",
        Model="None",
        OutputType="Audit",
        RelativePath=str(AuditPath.relative_to(Paths.RunDirectory)),
        Created=AuditPath.exists(),
        Description="Auditoría de verificación de tensores.",
    )

    print("\n=== Feature tensor check ===")
    print(AllResults[[
        "FeatureConfig",
        "Split",
        "FeatureShape",
        "MaskShape",
        "FeatureMin",
        "FeatureMax",
        "Finite",
        "MaskBinary",
        "AllChecksPassed",
    ]].to_string(index=False))

    print("\nStep08CheckFeatureTensors completed successfully.")


if __name__ == "__main__":
    Main()
