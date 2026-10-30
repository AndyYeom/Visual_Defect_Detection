import hashlib
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vdd.patchcore_inference import overlay_png, verify_file


def test_corrupt_or_missing_weights_are_rejected(tmp_path):
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'original weights')
    expected = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    verify_file(path, expected)
    path.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='integrity'):
        verify_file(path, expected)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        verify_file(path, expected)


def test_constant_map_and_non_square_image_produce_valid_overlay():
    image = Image.new('RGB', (173, 91), 'white')
    raw = overlay_png(image, np.ones((224, 224), dtype=np.float32))
    with Image.open(BytesIO(raw)) as overlay:
        assert overlay.size == image.size
        assert overlay.mode == 'RGB'
        assert overlay.format == 'PNG'
