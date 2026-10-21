import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vdd.data import load_manifest, make_manifest
from vdd.pipeline import Detector, evaluate, train
from vdd.scoring import calibrate, combine, metrics


def synthetic(root):
    script = Path(__file__).parents[1] / "scripts/make_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("synthetic", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate(root)


def test_end_to_end(tmp_path):
    root = tmp_path / "data"
    synthetic(root)
    manifest = tmp_path / "split.json"
    first = make_manifest(root, "synthetic", manifest)
    assert first == make_manifest(root, "synthetic", manifest)
    assert all(r["label"] == 0 for r in first["records"] if r["split"] != "test")
    train(root, manifest, tmp_path / "model", epochs=1, size=32, pretrained=False)
    checkpoint = tmp_path / "model/model.pt"
    report = evaluate(root, manifest, checkpoint, tmp_path / "report")
    assert set(report["metrics"]) == {"autoencoder", "cnn", "ensemble"}
    assert report["metrics"]["ensemble"]["n_images"] == 8
    with Image.open(root / "synthetic/test/good/000.png") as image:
        a, b = Detector(checkpoint).predict(image), Detector(checkpoint).predict(image)
    assert a["scores"] == b["scores"]
    assert a["heatmap"].shape == (32, 32)
    first["seed"] = 99
    manifest.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="differs"):
        evaluate(root, manifest, checkpoint, tmp_path / "bad-report")


def test_duplicates_and_tampering_rejected(tmp_path):
    synthetic(tmp_path)
    manifest = tmp_path / "split.json"
    make_manifest(tmp_path, "synthetic", manifest)
    source = tmp_path / "synthetic/train/good/000.png"
    source.write_bytes((tmp_path / "synthetic/test/good/000.png").read_bytes())
    with pytest.raises(ValueError, match="Changed|duplicate"):
        load_manifest(manifest, tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        make_manifest(tmp_path, "synthetic", manifest)


def test_calibration_and_metrics():
    scores = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    calibration = calibrate(scores)
    combined = combine(scores, calibration)
    np.testing.assert_allclose(combined[:, 0], combined[:, 1])
    np.testing.assert_allclose(combined[:, 2], combined[:, 0])
    assert np.isfinite(combine([[1, 1]], calibrate([[1, 1], [1, 1]]))).all()
    result = metrics([0, 1], [0.0, 1.0], 0.5)
    assert result["auroc"] == result["recall"] == result["precision"] == 1
    assert metrics([0, 0], [0.0, 1.0], 0.5)["auroc"] is None
    with pytest.raises(ValueError):
        calibrate(scores, weight=2)


def test_training_labels_rejected(tmp_path):
    synthetic(tmp_path)
    path = tmp_path / "split.json"
    manifest = make_manifest(tmp_path, "synthetic", path)
    manifest["records"][0]["label"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="normal-only"):
        load_manifest(path, tmp_path)
