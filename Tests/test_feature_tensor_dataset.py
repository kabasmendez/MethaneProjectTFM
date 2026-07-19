from pathlib import Path

import numpy as np

from Source.FeatureTensorDataset import FeatureTensorDataset


def test_feature_tensor_dataset_reads_sample(tmp_path: Path):
    features = np.ones((2, 3, 8, 8), dtype=np.float32)
    masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    masks[:, :, 2:4, 2:4] = 1

    feature_path = tmp_path / "Features.npy"
    mask_path = tmp_path / "Masks.npy"

    np.save(feature_path, features)
    np.save(mask_path, masks)

    dataset = FeatureTensorDataset(
        FeaturePath=feature_path,
        MaskPath=mask_path,
        ExpectedChannels=3,
        ExpectedHeight=8,
        ExpectedWidth=8,
    )

    assert len(dataset) == 2

    item = dataset[0]

    assert item["features"].shape == (3, 8, 8)
    assert item["mask"].shape == (1, 8, 8)
    assert item["features"].dtype.is_floating_point
    assert item["mask"].dtype.is_floating_point
