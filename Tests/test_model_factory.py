import torch

from Source.Models.ModelFactory import CreateModel, ListAvailableModels


def test_model_factory_lists_enhanced_unet():
    assert "EnhancedUNet" in ListAvailableModels()


def test_enhanced_unet_forward_configb_shape():
    model = CreateModel(
        ModelName="EnhancedUNet",
        InputChannels=9,
        OutputChannels=1,
        ModelParameters={
            "BaseChannels": 8,
            "ReflectPadding": 4,
            "UseSqueezeExcitation": True,
            "Dropout": 0.0,
        },
    )

    x = torch.randn(2, 9, 200, 200)
    y = model(x)

    assert y.shape == (2, 1, 200, 200)


def test_enhanced_unet_forward_configa_shape():
    model = CreateModel(
        ModelName="EnhancedUNet",
        InputChannels=7,
        OutputChannels=1,
        ModelParameters={
            "BaseChannels": 8,
            "ReflectPadding": 4,
            "UseSqueezeExcitation": True,
            "Dropout": 0.0,
        },
    )

    x = torch.randn(2, 7, 200, 200)
    y = model(x)

    assert y.shape == (2, 1, 200, 200)


def test_transformer_unet_forward_configb_shape():
    model = CreateModel(
        ModelName="TransformerUNet",
        InputChannels=9,
        OutputChannels=1,
        ModelParameters={
            "BaseChannels": 8,
            "ReflectPadding": 4,
            "UseSqueezeExcitation": True,
            "Dropout": 0.0,
            "TransformerHeads": 4,
            "TransformerLayers": 1,
            "TransformerMlpRatio": 2.0,
        },
    )

    x = torch.randn(2, 9, 200, 200)
    y = model(x)

    assert y.shape == (2, 1, 200, 200)
