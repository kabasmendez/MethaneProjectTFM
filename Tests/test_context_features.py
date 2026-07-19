import numpy as np
import pandas as pd

from Source.ContextFeatures import (
    BuildContextSummary,
    ComputeAngleSinCosDegrees,
    ComputeSolarAzimuthPolar,
    ComputeWindPolar,
    ExpandScalarToImage,
    FindContextColumnCandidates,
)


def test_find_context_columns():
    df = pd.DataFrame(
        {
            "meteo:wind_u": [1.0],
            "meteo:wind_v": [0.0],
            "satellite:saa": [90.0],
        }
    )

    candidates = FindContextColumnCandidates(df)

    assert candidates["WindU"] == "meteo:wind_u"
    assert candidates["WindV"] == "meteo:wind_v"
    assert candidates["SolarAzimuth"] == "satellite:saa"


def test_compute_wind_polar():
    values = ComputeWindPolar(1.0, 0.0)

    assert np.isclose(values["WindSpeed"], 1.0)
    assert np.isclose(values["WindCos"], 1.0)
    assert np.isclose(values["WindSin"], 0.0)


def test_compute_angle_sin_cos():
    values = ComputeAngleSinCosDegrees(90.0)

    assert np.isclose(values["Sin"], 1.0)
    assert np.isclose(values["Cos"], 0.0, atol=1e-6)


def test_compute_solar_azimuth():
    values = ComputeSolarAzimuthPolar(0.0)

    assert np.isclose(values["SolarAzimuthSin"], 0.0)
    assert np.isclose(values["SolarAzimuthCos"], 1.0)


def test_expand_scalar_to_image():
    image = ExpandScalarToImage(2.5, 4, 5)

    assert image.shape == (4, 5)
    assert np.isclose(image.mean(), 2.5)


def test_build_context_summary():
    df = pd.DataFrame(
        {
            "meteo:wind_u": [1.0, 2.0],
            "meteo:wind_v": [0.0, 0.5],
            "satellite:saa": [90.0, 100.0],
        }
    )

    summary = BuildContextSummary(df)

    assert len(summary) == 4
    assert summary["Exists"].sum() >= 3
