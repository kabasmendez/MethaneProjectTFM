from pathlib import Path

import numpy as np

from Source.ColorMaps import BuildNormalize, BuildRgbImage, GetFeatureStyle
from Source.VisualizationStyle import GetFigureSize, LoadVisualizationConfig


def test_visualization_config_loads():
    config = LoadVisualizationConfig(Path("Configs/VisualizationConfig.yaml"))
    assert "Visualization" in config
    assert "ColorMaps" in config
    assert "FeatureStyles" in config


def test_figure_size_exists():
    config = LoadVisualizationConfig(Path("Configs/VisualizationConfig.yaml"))
    width, height = GetFigureSize(config, "Single")
    assert width > 0
    assert height > 0


def test_feature_style_b11_exists():
    config = LoadVisualizationConfig(Path("Configs/VisualizationConfig.yaml"))
    style_name, style_config = GetFeatureStyle(config, "B11")
    assert style_name == "Reflectance"
    assert "Cmap" in style_config


def test_rgb_image_shape():
    red = np.random.rand(10, 10)
    green = np.random.rand(10, 10)
    blue = np.random.rand(10, 10)
    rgb = BuildRgbImage(red, green, blue)
    assert rgb.shape == (10, 10, 3)
    assert np.nanmin(rgb) >= 0
    assert np.nanmax(rgb) <= 1


def test_build_normalize_probability():
    config = LoadVisualizationConfig(Path("Configs/VisualizationConfig.yaml"))
    array = np.random.rand(10, 10)
    norm = BuildNormalize(array, config["ColorMaps"]["Probability"])
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0
