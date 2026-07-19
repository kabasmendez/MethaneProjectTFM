"""
FeatureEngineeringClean.py

Feature engineering limpio para ConfigB con MBMPPlus no supervisado.

Regla metodológica:
- Las features de entrada NO usan Plume.
- Las features de entrada NO usan ground truth.
- Plume se usa únicamente como máscara objetivo Y.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


EPSILON = 1e-6


def ToFloat32(Array: Any) -> np.ndarray:
    Array = np.asarray(Array, dtype=np.float32)
    return np.nan_to_num(Array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def SafeDivide(Numerator: Any, Denominator: Any, Epsilon: float = EPSILON) -> np.ndarray:
    Numerator = ToFloat32(Numerator)
    Denominator = ToFloat32(Denominator)
    return ToFloat32(Numerator / (Denominator + float(Epsilon)))


def GetBand(Image: Any, BandName: str) -> np.ndarray:
    """
    Extrae bandas desde array target/reference multibanda.

    Según el proyecto actual, target/reference llegan como array:
        13 x 200 x 200

    Índices Sentinel-2 esperados:
        B01=0, B02=1, B03=2, B04=3, B05=4, B06=5,
        B07=6, B08=7, B8A=8, B09=9, B10=10, B11=11, B12=12
    """
    Image = ToFloat32(Image)

    if Image.ndim != 3:
        raise ValueError(f"Image debe ser C x H x W. Shape recibido: {Image.shape}")

    BandIndex = {
        "B8A": 8,
        "B11": 11,
        "B12": 12,
    }

    if BandName not in BandIndex:
        raise KeyError(f"Banda no soportada: {BandName}")

    Index = BandIndex[BandName]

    if Image.shape[0] <= Index:
        raise ValueError(f"Image tiene {Image.shape[0]} canales, no alcanza para {BandName} index={Index}")

    return ToFloat32(Image[Index])


def ComputeNDSWIR(Target: Any) -> np.ndarray:
    B12 = GetBand(Target, "B12")
    B11 = GetBand(Target, "B11")
    return SafeDivide(B12 - B11, B12 + B11)


def ComputeRatio(A: Any, B: Any) -> np.ndarray:
    return SafeDivide(A, B)


def ComputeMBMP(TargetB12: Any, ReferenceB12: Any) -> np.ndarray:
    return SafeDivide(ToFloat32(TargetB12) - ToFloat32(ReferenceB12), ReferenceB12)


def RidgeClean2D(Image: Any, Sigma: float = 1.0) -> np.ndarray:
    Image = ToFloat32(Image)
    Smooth = gaussian_filter(Image, sigma=float(Sigma))
    return ToFloat32(Image - Smooth)


def ComputeMBMPPlus(TargetB12: Any, ReferenceB12: Any, Sigma: float = 1.0) -> np.ndarray:
    MBMP = ComputeMBMP(TargetB12, ReferenceB12)
    return RidgeClean2D(MBMP, Sigma=Sigma)


def ComputeDualEnhancementB12B11(Target: Any, Reference: Any) -> np.ndarray:
    B12Target = GetBand(Target, "B12")
    B11Target = GetBand(Target, "B11")
    B12Reference = GetBand(Reference, "B12")
    B11Reference = GetBand(Reference, "B11")

    TargetRatio = SafeDivide(B12Target, B11Target)
    ReferenceRatio = SafeDivide(B12Reference, B11Reference)

    return ToFloat32(SafeDivide(TargetRatio, ReferenceRatio) - 1.0)


def BuildConfigBFeatureDictionary(Target: Any, Reference: Any) -> dict[str, np.ndarray]:
    B8A = GetBand(Target, "B8A")
    B11 = GetBand(Target, "B11")
    B12 = GetBand(Target, "B12")
    ReferenceB12 = GetBand(Reference, "B12")

    FeatureDictionary = {
        "B8A": B8A,
        "B11": B11,
        "B12": B12,
        "NDSWIR": ComputeNDSWIR(Target),
        "RatioB12B11": ComputeRatio(B12, B11),
        "RatioB12B8A": ComputeRatio(B12, B8A),
        "MBMP": ComputeMBMP(B12, ReferenceB12),
        "MBMPPlus": ComputeMBMPPlus(B12, ReferenceB12),
        "DualEnhancementB12B11": ComputeDualEnhancementB12B11(Target, Reference),
    }

    return {Name: ToFloat32(Value) for Name, Value in FeatureDictionary.items()}


def BuildFeatureStack(Target: Any, Reference: Any, FeatureNames: list[str]) -> np.ndarray:
    FeatureDictionary = BuildConfigBFeatureDictionary(Target, Reference)

    Missing = [Name for Name in FeatureNames if Name not in FeatureDictionary]
    if Missing:
        raise KeyError(f"Features faltantes: {Missing}. Disponibles: {list(FeatureDictionary)}")

    Stack = np.stack([FeatureDictionary[Name] for Name in FeatureNames], axis=0)
    return ToFloat32(Stack)
