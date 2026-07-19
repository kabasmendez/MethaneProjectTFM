"""
FeatureEngineering.py

Cálculo auditado de features espectrales para MethaneProjectTFM.

ConfigA:
- B8A
- B11
- B12
- NDSWIR
- RatioB12B11
- RatioB12B8A
- MBMP

ConfigB:
- ConfigA
- MBMPPlus
- DualEnhancementB12B11

Decisión metodológica:
MBMPPlus se implementa como corrección supervisada de fondo siguiendo la
propuesta de César. Usa la máscara ground truth para definir píxeles de fondo
(plume == 0) y ajustar una regresión Ridge sobre esos píxeles.

Esta decisión está documentada en:
Docs/TechnicalDecision_MBMPPlus.md
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


S2_BAND_INDEX = {
    "B01": 0,
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}


CONFIG_A_FEATURES = [
    "B8A",
    "B11",
    "B12",
    "NDSWIR",
    "RatioB12B11",
    "RatioB12B8A",
    "MBMP",
]


CONFIG_B_FEATURES = CONFIG_A_FEATURES + [
    "MBMPPlus",
    "DualEnhancementB12B11",
]


def ValidateMultibandImage(Image: np.ndarray, Name: str) -> np.ndarray:
    """Valida imagen Sentinel-2 multibanda con forma Bands x H x W."""
    Array = np.asarray(Image, dtype=np.float32)

    if Array.ndim != 3:
        raise ValueError(f"{Name} debe tener forma (Bands, H, W). Recibido: {Array.shape}")

    if Array.shape[0] < 13:
        raise ValueError(f"{Name} debe tener al menos 13 bandas. Recibido: {Array.shape}")

    if not np.isfinite(Array).any():
        raise ValueError(f"{Name} no contiene valores finitos.")

    return Array


def ValidateMask(Mask: np.ndarray, ExpectedShape: tuple[int, int], Name: str = "PlumeMask") -> np.ndarray:
    """Valida máscara 2D."""
    MaskArray = np.asarray(Mask)

    if MaskArray.ndim != 2:
        raise ValueError(f"{Name} debe ser 2D. Recibido: {MaskArray.shape}")

    if MaskArray.shape != ExpectedShape:
        raise ValueError(f"{Name} debe tener forma {ExpectedShape}. Recibido: {MaskArray.shape}")

    return MaskArray > 0


def GetBand(Image: np.ndarray, BandName: str) -> np.ndarray:
    """Extrae una banda Sentinel-2 por nombre."""
    if BandName not in S2_BAND_INDEX:
        raise KeyError(f"Banda no reconocida: {BandName}")

    Image = ValidateMultibandImage(Image, "Image")
    return Image[S2_BAND_INDEX[BandName]].astype(np.float32)


def SafeDivide(Numerator: np.ndarray, Denominator: np.ndarray, Epsilon: float = 1e-6) -> np.ndarray:
    """División segura con epsilon."""
    Numerator = np.asarray(Numerator, dtype=np.float32)
    Denominator = np.asarray(Denominator, dtype=np.float32)

    return Numerator / (Denominator + Epsilon)


def ComputeNdswir(B12: np.ndarray, B11: np.ndarray, Epsilon: float = 1e-6) -> np.ndarray:
    """Calcula NDSWIR = (B12 - B11) / (B12 + B11 + eps)."""
    return SafeDivide(B12 - B11, B12 + B11, Epsilon=Epsilon).astype(np.float32)


def ComputeMbmpClassic(Target: np.ndarray, Reference: np.ndarray, Epsilon: float = 1e-6) -> np.ndarray:
    """
    Calcula MBMP clásico:

    MBMP = B12_target / B11_target - B12_reference / B11_reference
    """
    B12Target = GetBand(Target, "B12")
    B11Target = GetBand(Target, "B11")
    B12Reference = GetBand(Reference, "B12")
    B11Reference = GetBand(Reference, "B11")

    Mbmp = SafeDivide(B12Target, B11Target, Epsilon=Epsilon) - SafeDivide(
        B12Reference,
        B11Reference,
        Epsilon=Epsilon,
    )

    return Mbmp.astype(np.float32)


def ComputeMbmpInverse(Target: np.ndarray, Reference: np.ndarray, Epsilon: float = 1e-6) -> np.ndarray:
    """
    Calcula MBMP inverso, usado internamente por MBMPPlus:

    MBMP_inverse = B11_target / B12_target - B11_reference / B12_reference
    """
    B11Target = GetBand(Target, "B11")
    B12Target = GetBand(Target, "B12")
    B11Reference = GetBand(Reference, "B11")
    B12Reference = GetBand(Reference, "B12")

    MbmpInverse = SafeDivide(B11Target, B12Target, Epsilon=Epsilon) - SafeDivide(
        B11Reference,
        B12Reference,
        Epsilon=Epsilon,
    )

    return MbmpInverse.astype(np.float32)


def ComputeDualEnhancementB12B11(
    Target: np.ndarray,
    Reference: np.ndarray,
    Epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Calcula DualEnhancementB12B11:

    DualEnhancement = (B12_target / B11_target) /
                      (B12_reference / B11_reference + eps) - 1
    """
    B12Target = GetBand(Target, "B12")
    B11Target = GetBand(Target, "B11")
    B12Reference = GetBand(Reference, "B12")
    B11Reference = GetBand(Reference, "B11")

    TargetRatio = SafeDivide(B12Target, B11Target, Epsilon=Epsilon)
    ReferenceRatio = SafeDivide(B12Reference, B11Reference, Epsilon=Epsilon)

    DualEnhancement = SafeDivide(TargetRatio, ReferenceRatio, Epsilon=Epsilon) - 1.0

    return DualEnhancement.astype(np.float32)



def BuildFeatureDictionary(
    Target: np.ndarray,
    Reference: np.ndarray,
    PlumeMask: np.ndarray | None = None,
    FeatureConfig: str = "ConfigB",
    Epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Construye diccionario de features para ConfigA o ConfigB."""
    Target = ValidateMultibandImage(Target, "Target")
    Reference = ValidateMultibandImage(Reference, "Reference")

    if FeatureConfig not in ["ConfigA", "ConfigB"]:
        raise ValueError(f"FeatureConfig no soportada en esta fase: {FeatureConfig}")

    B8A = GetBand(Target, "B8A")
    B11 = GetBand(Target, "B11")
    B12 = GetBand(Target, "B12")

    FeatureDictionary = {
        "B8A": B8A,
        "B11": B11,
        "B12": B12,
        "NDSWIR": ComputeNdswir(B12, B11, Epsilon=Epsilon),
        "RatioB12B11": SafeDivide(B12, B11, Epsilon=Epsilon).astype(np.float32),
        "RatioB12B8A": SafeDivide(B12, B8A, Epsilon=Epsilon).astype(np.float32),
        "MBMP": ComputeMbmpClassic(Target, Reference, Epsilon=Epsilon),
    }

    if FeatureConfig == "ConfigB":
        if PlumeMask is None:
            raise ValueError("PlumeMask es obligatorio para ConfigB porque MBMPPlus lo requiere.")

        FeatureDictionary["MBMPPlus"] = ComputeMbmpPlus(
            Target=Target,
            Reference=Reference,
            PlumeMask=PlumeMask,
            RidgeAlpha=1.0,
            Epsilon=Epsilon,
        )
        FeatureDictionary["DualEnhancementB12B11"] = ComputeDualEnhancementB12B11(
            Target=Target,
            Reference=Reference,
            Epsilon=Epsilon,
        )

    ExpectedFeatures = CONFIG_A_FEATURES if FeatureConfig == "ConfigA" else CONFIG_B_FEATURES

    Missing = [Feature for Feature in ExpectedFeatures if Feature not in FeatureDictionary]
    if Missing:
        raise KeyError(f"Faltan features para {FeatureConfig}: {Missing}")

    return {Feature: FeatureDictionary[Feature] for Feature in ExpectedFeatures}


def SummarizeFeatureArray(
    FeatureName: str,
    Array: np.ndarray,
    PlumeMask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calcula estadísticas globales y plume/background."""
    Values = np.asarray(Array, dtype=np.float32)

    if Values.ndim != 2:
        raise ValueError(f"{FeatureName} debe ser 2D. Recibido: {Values.shape}")

    FiniteValues = Values[np.isfinite(Values)]

    if FiniteValues.size == 0:
        raise ValueError(f"{FeatureName} no contiene valores finitos.")

    Row: dict[str, Any] = {
        "Feature": FeatureName,
        "Shape": list(Values.shape),
        "FiniteCount": int(FiniteValues.size),
        "NanCount": int(np.isnan(Values).sum()),
        "Min": float(np.nanmin(Values)),
        "P02": float(np.nanpercentile(Values, 2)),
        "Mean": float(np.nanmean(Values)),
        "Median": float(np.nanmedian(Values)),
        "P98": float(np.nanpercentile(Values, 98)),
        "Max": float(np.nanmax(Values)),
        "Std": float(np.nanstd(Values)),
    }

    if PlumeMask is not None:
        Mask = ValidateMask(PlumeMask, ExpectedShape=Values.shape, Name="PlumeMask")

        PlumeValues = Values[Mask & np.isfinite(Values)]
        BackgroundValues = Values[(~Mask) & np.isfinite(Values)]

        Row["PlumePixelCount"] = int(PlumeValues.size)
        Row["BackgroundPixelCount"] = int(BackgroundValues.size)

        Row["PlumeMean"] = float(np.nanmean(PlumeValues)) if PlumeValues.size else np.nan
        Row["PlumeMedian"] = float(np.nanmedian(PlumeValues)) if PlumeValues.size else np.nan
        Row["BackgroundMean"] = (
            float(np.nanmean(BackgroundValues)) if BackgroundValues.size else np.nan
        )
        Row["BackgroundMedian"] = (
            float(np.nanmedian(BackgroundValues)) if BackgroundValues.size else np.nan
        )

        if PlumeValues.size and BackgroundValues.size:
            BackgroundStd = float(np.nanstd(BackgroundValues))
            Row["PlumeBackgroundDeltaMean"] = Row["PlumeMean"] - Row["BackgroundMean"]
            Row["ApproxSnr"] = (
                Row["PlumeBackgroundDeltaMean"] / BackgroundStd
                if BackgroundStd > 1e-12
                else np.nan
            )
        else:
            Row["PlumeBackgroundDeltaMean"] = np.nan
            Row["ApproxSnr"] = np.nan

    return Row


def BuildFeatureSummaryTable(
    FeatureDictionary: dict[str, np.ndarray],
    PlumeMask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Construye estadísticas para todas las features."""
    return [
        SummarizeFeatureArray(FeatureName, FeatureArray, PlumeMask=PlumeMask)
        for FeatureName, FeatureArray in FeatureDictionary.items()
    ]

def RidgeClean2D(Image, Sigma=1.0):
    """
    Limpieza espacial no supervisada aplicada al MBMP.

    No usa máscara de pluma, ground truth ni selección supervisada de fondo.
    """
    from scipy.ndimage import gaussian_filter

    Image = np.asarray(Image, dtype=np.float32)
    Smooth = gaussian_filter(Image, sigma=float(Sigma))
    Cleaned = Image - Smooth

    return np.nan_to_num(
        Cleaned,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def ComputeMbmpPlus(TargetB12, ReferenceB12, *args, Sigma=1.0, **kwargs):
    """
    MBMPPlus no supervisado.

    Decisión metodológica:
    - No usa Plume.
    - No usa GroundTruth.
    - No usa ValidBackground derivado de la máscara.
    - Ignora cualquier argumento extra que venga de código antiguo.

    Fórmula base:
        MBMP = (TargetB12 - ReferenceB12) / ReferenceB12

    Luego:
        MBMPPlus = RidgeClean2D(MBMP)
    """
    TargetB12 = np.asarray(TargetB12, dtype=np.float32)
    ReferenceB12 = np.asarray(ReferenceB12, dtype=np.float32)

    Epsilon = 1e-6

    MBMP = (TargetB12 - ReferenceB12) / (ReferenceB12 + Epsilon)
    MBMP = np.nan_to_num(
        MBMP,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    MBMPPlus = RidgeClean2D(MBMP, Sigma=Sigma)

    return MBMPPlus.astype(np.float32)


def ComputeMBMPPlus(TargetB12, ReferenceB12, *args, Sigma=1.0, **kwargs):
    """
    Alias compatible para MBMPPlus no supervisado.
    """
    return ComputeMbmpPlus(TargetB12, ReferenceB12, *args, Sigma=Sigma, **kwargs)

