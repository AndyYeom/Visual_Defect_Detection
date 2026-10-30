import numpy as np
import pytest
from PIL import Image

from vdd.benchmark import calibration, mask_for, preprocessing


def test_normal_calibration_rejects_invalid():
    with pytest.raises(ValueError):
        calibration([np.nan])
    assert calibration([1, 2, 3])["threshold"] == pytest.approx(2.9)


def test_mask_uses_defect_type_and_nearest_resize(tmp_path):
    folder = tmp_path / 'bottle/ground_truth/crack'
    folder.mkdir(parents=True)
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8)).save(folder / '001_mask.png')
    mask = mask_for(tmp_path, {'path': 'bottle/test/crack/001.png', 'label': 1}, 4)
    assert mask.sum() == 4
    assert not mask_for(tmp_path, {'label': 0}, 4).any()
    assert preprocessing(224, 'anomalydino')(Image.new('RGB', (64, 64))).shape == (3, 224, 224)
