import numpy as np
import pytest
import torch
from PIL import Image
from test_pipeline import synthetic

from vdd.data import make_manifest
from vdd.device import resolve_device
from vdd.pipeline import Detector, evaluate, train


def test_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(resolve_device()) == "cpu"
    for device in ["cuda", "mps", "invalid"]:
        with pytest.raises(ValueError):
            resolve_device(device)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert str(resolve_device()) == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(resolve_device()) == "cuda"
    assert str(resolve_device("cpu")) == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_checkpoint_portability(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    synthetic(tmp_path / "data")
    root = tmp_path / "data"
    manifest = tmp_path / "split.json"
    make_manifest(root, "synthetic", manifest)
    train(root, manifest, tmp_path / "model", epochs=1, size=32,
          pretrained=False, device=device)
    checkpoint = tmp_path / "model/model.pt"
    state = torch.load(checkpoint, weights_only=True)
    assert all(t.device.type == "cpu" for t in state["ae"].values())
    assert state["mean"].device.type == "cpu"
    with Image.open(root / "synthetic/test/scratch/000.png") as image:
        accelerated = Detector(checkpoint, device=device).predict(image)
        cpu = Detector(checkpoint, device="cpu").predict(image)
    np.testing.assert_allclose(list(accelerated["scores"].values()),
                               list(cpu["scores"].values()), rtol=0.02, atol=0.02)
    assert np.isfinite(accelerated["heatmap"]).all()
    report = evaluate(root, manifest, checkpoint, tmp_path / "report", device=device)
    assert report["device"] == device
