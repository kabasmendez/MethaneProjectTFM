"""
FeatureEngineering.py

Construcción de features para MethaneProjectTFM.

Decisión metodológica actual:
- ConfigB usa MBMPPlus no supervisado.
- Las features de entrada no usan Plume.
- Las features de entrada no usan GroundTruth.
- Plume se usa únicamente como máscara objetivo Y.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


EPSILON = 1e-6

BAND_INDEX = {
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


def ToFloat32(Array: Any) -> np.ndarray:
    Array = np.asarray(Array, dtype=np.float32)
    return np.nan_to_num(Array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def SafeDivide(Numerator: Any, Denominator: Any, Epsilon: float = EPSILON) -> np.ndarray:
    Numerator = ToFloat32(Numerator)
    Denominator = ToFloat32(Denominator)
    Result = Numerator / (Denominator + float(Epsilon))
    return ToFloat32(Result)


def GetBand(ImageOrDict: Any, BandName: str) -> np.ndarray:
    """
    Extrae banda desde:
    - array C x H x W
    - diccionario con llaves tipo B8A, B11, B12
    """
    if isinstance(ImageOrDict, dict):
        Aliases = {
            "B8A": ["B8A", "B08A", "B8"],
            "B11": ["B11"],
            "B12": ["B12"],
        }

        for Key in Aliases.get(BandName, [BandName]):
            if Key in ImageOrDict:
                return ToFloat32(ImageOrDict[Key])

        raise KeyError(f"No encontré banda {BandName}. Disponibles: {list(ImageOrDict.keys())}")

    Image = ToFloat32(ImageOrDict)

    if Image.ndim != 3:
        raise ValueError(f"Se esperaba imagen C x H x W para extraer {BandName}. Shape recibido: {Image.shape}")

    if BandName not in BAND_INDEX:
        raise KeyError(f"Banda no soportada: {BandName}")

    Index = BAND_INDEX[BandName]

    if Image.shape[0] <= Index:
        raise ValueError(f"La imagen tiene {Image.shape[0]} canales; no alcanza para {BandName} index={Index}.")

    return ToFloat32(Image[Index])


def ComputeNormalizedDifference(A: Any, B: Any) -> np.ndarray:
    A = ToFloat32(A)
    B = ToFloat32(B)
    return SafeDivide(A - B, A + B)


def ComputeRatio(A: Any, B: Any) -> np.ndarray:
    return SafeDivide(A, B)


def ComputeMbmp(TargetB12: Any, ReferenceB12: Any) -> np.ndarray:
    """
    MBMP clásico:

        MBMP = (TargetB12 - ReferenceB12) / ReferenceB12
    """
    TargetB12 = ToFloat32(TargetB12)
    ReferenceB12 = ToFloat32(ReferenceB12)
    return SafeDivide(TargetB12 - ReferenceB12, ReferenceB12)


def ComputeMBMP(TargetB12: Any, ReferenceB12: Any) -> np.ndarray:
    return ComputeMbmp(TargetB12, ReferenceB12)


def RidgeClean2D(Image: Any, Sigma: float = 1.0) -> np.ndarray:
    """
    Limpieza espacial no supervisada aplicada a MBMP.

    Usa solo la imagen MBMP; no usa etiquetas ni máscaras.
    """
    Image = ToFloat32(Image)
    Smooth = gaussian_filter(Image, sigma=float(Sigma))
    return ToFloat32(Image - Smooth)


def ComputeMbmpPlus(TargetB12: Any, ReferenceB12: Any, *args: Any, Sigma: float = 1.0, **kwargs: Any) -> np.ndarray:
    """
    MBMPPlus no supervisado.

    Acepta argumentos extra para compatibilidad con código antiguo,
    pero los ignora deliberadamente.
    """
    MBMP = ComputeMbmp(TargetB12, ReferenceB12)
    return RidgeClean2D(MBMP, Sigma=Sigma)


def ComputeMBMPPlus(TargetB12: Any, ReferenceB12: Any, *args: Any, Sigma: float = 1.0, **kwargs: Any) -> np.ndarray:
    return ComputeMbmpPlus(TargetB12, ReferenceB12, *args, Sigma=Sigma, **kwargs)


def ComputeNDSWIR(Target: Any) -> np.ndarray:
    B12 = GetBand(Target, "B12")
    B11 = GetBand(Target, "B11")
    return ComputeNormalizedDifference(B12, B11)


def ComputeDualEnhancementB12B11(Target: Any, Reference: Any) -> np.ndarray:
    B12Target = GetBand(Target, "B12")
    B11Target = GetBand(Target, "B11")
    B12Reference = GetBand(Reference, "B12")
    B11Reference = GetBand(Reference, "B11")

    TargetRatio = SafeDivide(B12Target, B11Target)
    ReferenceRatio = SafeDivide(B12Reference, B11Reference)

    return ToFloat32(SafeDivide(TargetRatio, ReferenceRatio) - 1.0)



def BuildWindFeatureChannelsFromMetadata(
    ContextMetadata: dict[str, Any] | None,
    Shape: tuple[int, int],
    Eps: float = EPSILON,
) -> dict[str, np.ndarray]:
    """
    Construye canales constantes HxW a partir de metadatos de viento.

    Entradas esperadas:
    - meteo:wind_u
    - meteo:wind_v

    Salidas:
    - WindSpeed10m
    - WindDirCos10m
    - WindDirSin10m

    No usa PlumeMask ni GroundTruth.
    """
    if ContextMetadata is None:
        raise ValueError(
            "ConfigC requiere ContextMetadata con 'meteo:wind_u' y 'meteo:wind_v'."
        )

    WindUKeys = ["meteo:wind_u", "wind_u", "WindU", "meteo.wind_u"]
    WindVKeys = ["meteo:wind_v", "wind_v", "WindV", "meteo.wind_v"]

    WindU = None
    WindV = None

    for Key in WindUKeys:
        if Key in ContextMetadata:
            WindU = ContextMetadata[Key]
            break

    for Key in WindVKeys:
        if Key in ContextMetadata:
            WindV = ContextMetadata[Key]
            break

    if WindU is None or WindV is None:
        raise KeyError(
            "No se encontraron columnas de viento en ContextMetadata. "
            f"Claves disponibles: {list(ContextMetadata.keys())[:80]}"
        )

    WindU = float(WindU)
    WindV = float(WindV)

    WindSpeed = float(np.sqrt(WindU ** 2 + WindV ** 2))
    WindDirCos = float(WindU / (WindSpeed + float(Eps)))
    WindDirSin = float(WindV / (WindSpeed + float(Eps)))

    Height, Width = tuple(Shape)

    return {
        "WindSpeed10m": np.full((Height, Width), WindSpeed, dtype=np.float32),
        "WindDirCos10m": np.full((Height, Width), WindDirCos, dtype=np.float32),
        "WindDirSin10m": np.full((Height, Width), WindDirSin, dtype=np.float32),
    }

def BuildFeatureDictionary(
    Target: Any,
    Reference: Any,
    FeatureConfig: str | None = None,
    PlumeMask: np.ndarray | None = None,
    ContextMetadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """
    Construye features disponibles para ConfigA/ConfigB/ConfigC.

    PlumeMask se acepta por compatibilidad, pero no se usa para construir features.

    ConfigB:
    - Features espectrales + MBMPPlus no supervisado.

    ConfigC:
    - ConfigB + WindSpeed10m + WindDirCos10m + WindDirSin10m.
    - Las variables de viento se calculan desde meteo:wind_u/meteo:wind_v.
    - Se rasterizan como canales constantes HxW.
    """
    # Compatibilidad: algunos scripts pueden pasar Metadata=...
    if ContextMetadata is None and "Metadata" in kwargs:
        ContextMetadata = kwargs["Metadata"]

    B8A = GetBand(Target, "B8A")
    B11 = GetBand(Target, "B11")
    B12 = GetBand(Target, "B12")
    ReferenceB12 = GetBand(Reference, "B12")

    Features = {
        "B8A": B8A,
        "B11": B11,
        "B12": B12,
        "NDSWIR": ComputeNDSWIR(Target),
        "RatioB12B11": ComputeRatio(B12, B11),
        "RatioB12B8A": ComputeRatio(B12, B8A),
        "MBMP": ComputeMbmp(B12, ReferenceB12),
        "MBMPPlus": ComputeMbmpPlus(B12, ReferenceB12),
        "DualEnhancementB12B11": ComputeDualEnhancementB12B11(Target, Reference),
    }

    if str(FeatureConfig) == "ConfigC":
        WindFeatures = BuildWindFeatureChannelsFromMetadata(
            ContextMetadata=ContextMetadata,
            Shape=B12.shape,
        )
        Features.update(WindFeatures)

    return {Name: ToFloat32(Value) for Name, Value in Features.items()}


def BuildFeatureStack(
    Target: Any,
    Reference: Any,
    FeatureNames: list[str],
    FeatureConfig: str | None = None,
    PlumeMask: np.ndarray | None = None,
    ContextMetadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    # Compatibilidad: permitir Metadata=... como alias de ContextMetadata.
    if ContextMetadata is None and "Metadata" in kwargs:
        ContextMetadata = kwargs["Metadata"]

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
        raise KeyError(f"Features faltantes: {Missing}. Disponibles: {list(FeatureDictionary.keys())}")

    Stack = np.stack([FeatureDictionary[Name] for Name in FeatureNames], axis=0)
    return ToFloat32(Stack)


def ValidateMask(Mask: Any, ExpectedShape: tuple[int, int] | None = None, Name: str = "Mask") -> np.ndarray:
    Mask = np.asarray(Mask)

    if Mask.ndim != 2:
        raise ValueError(f"{Name} debe ser 2D. Shape recibido: {Mask.shape}")

    if ExpectedShape is not None and tuple(Mask.shape) != tuple(ExpectedShape):
        raise ValueError(f"{Name} debe tener shape {ExpectedShape}. Recibido: {Mask.shape}")

    return Mask > 0


def SummarizeFeatureArray(
    FeatureName: str,
    FeatureArray: np.ndarray,
    PlumeMask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Resumen estadístico de una feature.

    PlumeMask solo se usa para análisis descriptivo, no para construir features.
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
            Delta = Row["PlumeMean"] - Row["BackgroundMean"]
            Row["PlumeBackgroundDeltaMean"] = Delta
            Row["PlumeBackgroundDeltaZ"] = Delta / BackgroundStd if BackgroundStd > 0 else np.nan
        else:
            Row["PlumeBackgroundDeltaMean"] = np.nan
            Row["PlumeBackgroundDeltaZ"] = np.nan

    return Row


def BuildFeatureSummaryTable(
    FeatureDictionary: dict[str, np.ndarray],
    PlumeMask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    return [
        SummarizeFeatureArray(Name, Array, PlumeMask=PlumeMask)
        for Name, Array in FeatureDictionary.items()
    ]
