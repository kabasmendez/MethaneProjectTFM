"""
FeatureEngineering.py

Construcción de features para MethaneProjectTFM.

Decisión metodológica:
- MBMPPlus se calcula de forma no supervisada.
- No usa máscara de pluma.
- No usa ground truth.
- No ajusta fondo usando etiquetas.
- Es reproducible en inferencia operacional.

ConfigB:
    B8A, B11, B12, NDSWIR, RatioB12B11, RatioB12B8A,
    MBMP, MBMPPlus, DualEnhancementB12B11
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


EPSILON = 1e-6


DEFAULT_FEATURES = [
    "B8A",
    "B11",
    "B12",
    "NDSWIR",
    "RatioB12B11",
    "RatioB12B8A",
    "MBMP",
    "MBMPPlus",
    "DualEnhancementB12B11",
]


def ToFloat32(Array: Any) -> np.ndarray:
    """
    Convierte a float32 y reemplaza NaN/Inf por cero.
    """
    Output = np.asarray(Array, dtype=np.float32)

    return np.nan_to_num(
        Output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def ValidateImage(Image: Any, ExpectedShape: tuple[int, int] | None = None, Name: str = "Image") -> np.ndarray:
    """
    Valida imagen 2D.
    """
    Image = ToFloat32(Image)

    if Image.ndim != 2:
        raise ValueError(f"{Name} debe ser 2D. Shape recibido: {Image.shape}")

    if ExpectedShape is not None and tuple(Image.shape) != tuple(ExpectedShape):
        raise ValueError(f"{Name} debe tener shape {ExpectedShape}. Recibido: {Image.shape}")

    return Image


def ValidateMask(Mask: Any, ExpectedShape: tuple[int, int] | None = None, Name: str = "Mask") -> np.ndarray:
    """
    Valida máscara 2D. Se mantiene para compatibilidad con resúmenes/visualización.

    Esta función NO se usa para calcular MBMPPlus.
    """
    Mask = np.asarray(Mask)

    if Mask.ndim != 2:
        raise ValueError(f"{Name} debe ser 2D. Shape recibido: {Mask.shape}")

    if ExpectedShape is not None and tuple(Mask.shape) != tuple(ExpectedShape):
        raise ValueError(f"{Name} debe tener shape {ExpectedShape}. Recibido: {Mask.shape}")

    return Mask > 0


def GetBand(Dictionary: dict[str, Any], Name: str) -> np.ndarray:
    """
    Obtiene una banda desde un diccionario.

    Acepta alias frecuentes para B8A.
    """
    Aliases = {
        "B8A": ["B8A", "B08A", "B8"],
        "B08A": ["B08A", "B8A", "B8"],
        "B8": ["B8", "B8A", "B08A"],
        "B11": ["B11"],
        "B12": ["B12"],
    }

    Candidates = Aliases.get(Name, [Name])

    for Candidate in Candidates:
        if Candidate in Dictionary:
            return ValidateImage(Dictionary[Candidate], Name=Candidate)

    Available = ", ".join(sorted(Dictionary.keys()))
    raise KeyError(f"No se encontró banda {Name}. Disponibles: {Available}")


def SafeDivide(Numerator: Any, Denominator: Any, Epsilon: float = EPSILON) -> np.ndarray:
    """
    División estable.
    """
    Numerator = ToFloat32(Numerator)
    Denominator = ToFloat32(Denominator)

    Result = Numerator / (Denominator + float(Epsilon))

    return ToFloat32(Result)


def ComputeNormalizedDifference(A: Any, B: Any, Epsilon: float = EPSILON) -> np.ndarray:
    """
    Índice normalizado: (A - B) / (A + B).
    """
    A = ToFloat32(A)
    B = ToFloat32(B)

    return SafeDivide(A - B, A + B, Epsilon=Epsilon)


def ComputeRatio(A: Any, B: Any, Epsilon: float = EPSILON) -> np.ndarray:
    """
    Razón espectral A / B.
    """
    return SafeDivide(A, B, Epsilon=Epsilon)


def ComputeNDSWIR(Target: dict[str, Any], Epsilon: float = EPSILON) -> np.ndarray:
    """
    NDSWIR = (B12 - B11) / (B12 + B11)
    """
    B12 = GetBand(Target, "B12")
    B11 = GetBand(Target, "B11")

    return ComputeNormalizedDifference(B12, B11, Epsilon=Epsilon)


def ComputeMbmp(TargetB12: Any, ReferenceB12: Any, Epsilon: float = EPSILON) -> np.ndarray:
    """
    MBMP clásico:

        MBMP = (TargetB12 - ReferenceB12) / ReferenceB12
    """
    TargetB12 = ToFloat32(TargetB12)
    ReferenceB12 = ToFloat32(ReferenceB12)

    return SafeDivide(TargetB12 - ReferenceB12, ReferenceB12, Epsilon=Epsilon)


def ComputeMBMP(TargetB12: Any, ReferenceB12: Any, Epsilon: float = EPSILON) -> np.ndarray:
    """
    Alias compatible.
    """
    return ComputeMbmp(TargetB12, ReferenceB12, Epsilon=Epsilon)


def RidgeClean2D(Image: Any, Sigma: float = 1.0) -> np.ndarray:
    """
    Limpieza espacial no supervisada aplicada al MBMP.

    Usa únicamente la imagen de entrada.
    """
    Image = ToFloat32(Image)

    Smooth = gaussian_filter(Image, sigma=float(Sigma))
    Cleaned = Image - Smooth

    return ToFloat32(Cleaned)


def ComputeMbmpPlus(TargetB12: Any, ReferenceB12: Any, *args: Any, Sigma: float = 1.0, **kwargs: Any) -> np.ndarray:
    """
    MBMPPlus no supervisado.

    Se aceptan argumentos extra para compatibilidad con llamadas antiguas,
    pero se ignoran deliberadamente.
    """
    MBMP = ComputeMbmp(TargetB12, ReferenceB12)
    MBMPPlus = RidgeClean2D(MBMP, Sigma=Sigma)

    return ToFloat32(MBMPPlus)


def ComputeMBMPPlus(TargetB12: Any, ReferenceB12: Any, *args: Any, Sigma: float = 1.0, **kwargs: Any) -> np.ndarray:
    """
    Alias compatible.
    """
    return ComputeMbmpPlus(TargetB12, ReferenceB12, *args, Sigma=Sigma, **kwargs)


def ComputeDualEnhancementB12B11(
    Target: dict[str, Any],
    Reference: dict[str, Any],
    Epsilon: float = EPSILON,
) -> np.ndarray:
    """
    DualEnhancementB12B11:

        (B12_target / B11_target) / (B12_reference / B11_reference) - 1
    """
    B12Target = GetBand(Target, "B12")
    B11Target = GetBand(Target, "B11")
    B12Reference = GetBand(Reference, "B12")
    B11Reference = GetBand(Reference, "B11")

    TargetRatio = SafeDivide(B12Target, B11Target, Epsilon=Epsilon)
    ReferenceRatio = SafeDivide(B12Reference, B11Reference, Epsilon=Epsilon)

    DualEnhancement = SafeDivide(TargetRatio, ReferenceRatio, Epsilon=Epsilon) - 1.0

    return ToFloat32(DualEnhancement)


def BuildFeatureDictionary(
    Target: dict[str, Any],
    Reference: dict[str, Any],
    FeatureConfig: str | None = None,
    PlumeMask: np.ndarray | None = None,
    ContextMetadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """
    Construye el diccionario completo de features.

    PlumeMask se acepta por compatibilidad con Step07, pero NO se usa para
    construir ninguna feature.
    """
    B8A = GetBand(Target, "B8A")
    B11 = GetBand(Target, "B11")
    B12 = GetBand(Target, "B12")

    ReferenceB12 = GetBand(Reference, "B12")

    FeatureDictionary: dict[str, np.ndarray] = {}

    FeatureDictionary["B8A"] = B8A
    FeatureDictionary["B11"] = B11
    FeatureDictionary["B12"] = B12

    FeatureDictionary["NDSWIR"] = ComputeNDSWIR(Target)
    FeatureDictionary["RatioB12B11"] = ComputeRatio(B12, B11)
    FeatureDictionary["RatioB12B8A"] = ComputeRatio(B12, B8A)

    FeatureDictionary["MBMP"] = ComputeMbmp(B12, ReferenceB12)
    FeatureDictionary["MBMPPlus"] = ComputeMbmpPlus(B12, ReferenceB12)

    FeatureDictionary["DualEnhancementB12B11"] = ComputeDualEnhancementB12B11(
        Target=Target,
        Reference=Reference,
    )

    for Name, Value in list(FeatureDictionary.items()):
        FeatureDictionary[Name] = ToFloat32(Value)

    return FeatureDictionary


def BuildSelectedFeatureStack(
    Target: dict[str, Any],
    Reference: dict[str, Any],
    FeatureNames: list[str],
    FeatureConfig: str | None = None,
    PlumeMask: np.ndarray | None = None,
    ContextMetadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Construye stack C x H x W a partir de nombres de features.
    """
    FeatureDictionary = BuildFeatureDictionary(
        Target=Target,
        Reference=Reference,
        FeatureConfig=FeatureConfig,
        PlumeMask=PlumeMask,
        ContextMetadata=ContextMetadata,
        **kwargs,
    )

    Missing = [Name for Name in FeatureNames if Name not in FeatureDictionary]
    if Missing:
        Available = ", ".join(sorted(FeatureDictionary.keys()))
        raise KeyError(f"Features no disponibles: {Missing}. Disponibles: {Available}")

    Stack = np.stack([FeatureDictionary[Name] for Name in FeatureNames], axis=0)

    return ToFloat32(Stack)


def SummarizeFeatureArray(
    FeatureName: str,
    FeatureArray: np.ndarray,
    PlumeMask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Resume una feature.

    Si se entrega PlumeMask, solo se usa para resumen estadístico, no para
    construir features.
    """
    Values = ToFloat32(FeatureArray)

    Row: dict[str, Any] = {
        "Feature": FeatureName,
        "Shape": str(tuple(Values.shape)),
        "Dtype": str(Values.dtype),
        "FinitePixels": int(np.isfinite(Values).sum()),
        "NaNPixels": int(np.isnan(Values).sum()),
        "Min": float(np.nanmin(Values)),
        "Max": float(np.nanmax(Values)),
        "Mean": float(np.nanmean(Values)),
        "Std": float(np.nanstd(Values)),
        "P01": float(np.nanpercentile(Values, 1)),
        "P05": float(np.nanpercentile(Values, 5)),
        "P50": float(np.nanpercentile(Values, 50)),
        "P95": float(np.nanpercentile(Values, 95)),
        "P99": float(np.nanpercentile(Values, 99)),
    }

    if PlumeMask is not None:
        Mask = ValidateMask(PlumeMask, ExpectedShape=Values.shape, Name="PlumeMask")
        BackgroundMask = ~Mask

        PlumeValues = Values[Mask & np.isfinite(Values)]
        BackgroundValues = Values[BackgroundMask & np.isfinite(Values)]

        Row["PlumePixelCount"] = int(PlumeValues.size)
        Row["BackgroundPixelCount"] = int(BackgroundValues.size)

        Row["PlumeMean"] = float(np.nanmean(PlumeValues)) if PlumeValues.size else np.nan
        Row["BackgroundMean"] = float(np.nanmean(BackgroundValues)) if BackgroundValues.size else np.nan
        Row["PlumeMedian"] = float(np.nanmedian(PlumeValues)) if PlumeValues.size else np.nan
        Row["BackgroundMedian"] = float(np.nanmedian(BackgroundValues)) if BackgroundValues.size else np.nan

        if PlumeValues.size and BackgroundValues.size:
            BackgroundStd = float(np.nanstd(BackgroundValues))
            Row["PlumeBackgroundDeltaMean"] = Row["PlumeMean"] - Row["BackgroundMean"]
            Row["PlumeBackgroundDeltaZ"] = (
                Row["PlumeBackgroundDeltaMean"] / BackgroundStd
                if BackgroundStd > 0
                else np.nan
            )
        else:
            Row["PlumeBackgroundDeltaMean"] = np.nan
            Row["PlumeBackgroundDeltaZ"] = np.nan

    return Row


def BuildFeatureSummaryTable(
    FeatureDictionary: dict[str, np.ndarray],
    PlumeMask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Construye tabla de resumen de features.
    """
    Rows = []

    for FeatureName, FeatureArray in FeatureDictionary.items():
        Rows.append(
            SummarizeFeatureArray(
                FeatureName=FeatureName,
                FeatureArray=FeatureArray,
                PlumeMask=PlumeMask,
            )
        )

    return Rows
