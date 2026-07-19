import numpy as np
import pytest

from Source.FeatureEngineering import (
    CONFIG_A_FEATURES,
    CONFIG_B_FEATURES,
    BuildFeatureDictionary,
    ComputeMbmpClassic,
    ComputeMbmpPlus,
    ComputeNdswir,
    GetBand,
    SafeDivide,
)


def test_safe_divide_shape_and_finite():
    a = np.ones((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    out = SafeDivide(a, b)
    assert out.shape == (4, 4)
    assert np.isfinite(out).all()


def test_get_band_b12():
    image = np.zeros((13, 5, 5), dtype=np.float32)
    image[12] = 7.0
    band = GetBand(image, "B12")
    assert band.shape == (5, 5)
    assert np.allclose(band, 7.0)


def test_ndswir_simple_value():
    b12 = np.full((4, 4), 3.0, dtype=np.float32)
    b11 = np.full((4, 4), 1.0, dtype=np.float32)
    out = ComputeNdswir(b12, b11)
    assert np.allclose(out, 0.5, atol=1e-5)


def test_mbmp_classic_zero_when_target_equals_reference():
    target = np.ones((13, 6, 6), dtype=np.float32)
    reference = np.ones((13, 6, 6), dtype=np.float32)
    out = ComputeMbmpClassic(target, reference)
    assert out.shape == (6, 6)
    assert np.allclose(out, 0.0, atol=1e-5)


def test_mbmp_plus_shape_and_finite():
    rng = np.random.default_rng(42)
    target = rng.normal(1.0, 0.05, size=(13, 30, 30)).astype(np.float32)
    reference = rng.normal(1.0, 0.05, size=(13, 30, 30)).astype(np.float32)

    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[10:15, 10:15] = 1

    out = ComputeMbmpPlus(target, reference, mask)

    assert out.shape == (30, 30)
    assert np.isfinite(out).all()


def test_build_configa_dictionary_has_7_features():
    rng = np.random.default_rng(1)
    target = rng.normal(1.0, 0.05, size=(13, 20, 20)).astype(np.float32)
    reference = rng.normal(1.0, 0.05, size=(13, 20, 20)).astype(np.float32)

    features = BuildFeatureDictionary(
        Target=target,
        Reference=reference,
        PlumeMask=None,
        FeatureConfig="ConfigA",
    )

    assert list(features.keys()) == CONFIG_A_FEATURES
    assert len(features) == 7


def test_build_configb_dictionary_has_9_features():
    rng = np.random.default_rng(2)
    target = rng.normal(1.0, 0.05, size=(13, 30, 30)).astype(np.float32)
    reference = rng.normal(1.0, 0.05, size=(13, 30, 30)).astype(np.float32)

    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[12:18, 12:18] = 1

    features = BuildFeatureDictionary(
        Target=target,
        Reference=reference,
        PlumeMask=mask,
        FeatureConfig="ConfigB",
    )

    assert list(features.keys()) == CONFIG_B_FEATURES
    assert len(features) == 9

    for array in features.values():
        assert array.shape == (30, 30)
        assert np.isfinite(array).all()


def test_configb_requires_plume_mask():
    target = np.ones((13, 20, 20), dtype=np.float32)
    reference = np.ones((13, 20, 20), dtype=np.float32)

    with pytest.raises(ValueError):
        BuildFeatureDictionary(
            Target=target,
            Reference=reference,
            PlumeMask=None,
            FeatureConfig="ConfigB",
        )
