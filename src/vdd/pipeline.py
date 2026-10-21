"""CPU-first reproducible baseline training, inference, and hold-out evaluation."""

import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import ImageDataset, load_manifest, transform
from .models import Autoencoder, FeatureModel, anomaly_maps, image_score
from .scoring import calibrate, combine, metrics


def train(
    root,
    manifest_path,
    output,
    epochs=10,
    batch_size=16,
    size=128,
    seed=42,
    weight=0.5,
    quantile=0.95,
    pretrained=True,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    transform(size)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    manifest = load_manifest(manifest_path, root)

    def loader(split, shuffle=False):
        records = [r for r in manifest["records"] if r["split"] == split]
        return DataLoader(ImageDataset(root, records, size), batch_size=batch_size, shuffle=shuffle)

    train_loader, val_loader = loader("train", True), loader("val")
    ae, cnn = Autoencoder(), FeatureModel(pretrained=pretrained)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        ae.train()
        total, count = 0.0, 0
        for x, _, _ in train_loader:
            optimizer.zero_grad()
            loss = (ae(x) - x).square().mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        history.append({"epoch": epoch + 1, "train_mse": total / count})
    ae.eval()
    with torch.inference_mode():
        total, square, count = None, None, 0
        for x, _, _ in train_loader:
            features = cnn(x).double()
            s, ss = features.sum(0, keepdim=True), features.square().sum(0, keepdim=True)
            total = s if total is None else total + s
            square = ss if square is None else square + ss
            count += len(x)
        mean = total / count
        variance = (square / count - mean.square()).clamp_min(1e-4).float()
        mean = mean.float()
        scores = []
        for x, _, _ in val_loader:
            maps = anomaly_maps(ae, cnn, x, mean, variance)
            scores.extend(torch.stack([image_score(m) for m in maps], 1).tolist())
    calibration = calibrate(scores, weight, quantile)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "category": manifest["category"],
        "size": size,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "pretrained": pretrained,
        "validation_count": len(scores),
        "calibration": calibration,
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "history": history,
    }
    torch.save(
        {
            "ae": ae.state_dict(),
            "cnn": cnn.state_dict(),
            "mean": mean,
            "variance": variance,
            "metadata": metadata,
        },
        output / "model.pt",
    )
    (output / "training.json").write_text(json.dumps(metadata, indent=2))
    return metadata


class Detector:
    def __init__(self, checkpoint):
        torch.set_num_threads(2)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.metadata = state["metadata"]
        self.ae, self.cnn = Autoencoder().eval(), FeatureModel(pretrained=False)
        self.ae.load_state_dict(state["ae"])
        self.cnn.load_state_dict(state["cnn"])
        self.mean, self.variance = state["mean"], state["variance"]
        self.transform = transform(self.metadata["size"])

    @torch.inference_mode()
    def predict(self, image):
        start = time.perf_counter()
        x = self.transform(image.convert("RGB"))[None]
        t0 = time.perf_counter()
        ae_map = (self.ae(x) - x).square().mean(1)
        ae_raw = image_score(ae_map).item()
        t1 = time.perf_counter()
        from torch.nn import functional as F

        fmap = ((self.cnn(x) - self.mean).square() / self.variance).mean(1)
        cnn_map = F.interpolate(
            fmap[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        cnn_raw = image_score(cnn_map).item()
        t2 = time.perf_counter()
        calibration = self.metadata["calibration"]
        scores = combine([[ae_raw, cnn_raw]], calibration)[0]
        # Display only: image-level calibration is applied to maps for an overlay.
        # This heatmap is not a calibrated pixel-level classifier.
        w = calibration["weight"]
        heatmap = (
            w * (ae_map[0].numpy() - calibration["center"][0]) / calibration["scale"][0]
            + (1 - w) * (cnn_map[0].numpy() - calibration["center"][1]) / calibration["scale"][1]
        )
        end = time.perf_counter()
        return {
            "scores": dict(zip(("autoencoder", "cnn", "ensemble"), scores.tolist())),
            "anomalous": bool(scores[2] > calibration["thresholds"][2]),
            "threshold": calibration["thresholds"][2],
            "heatmap": heatmap,
            "timing_ms": {
                "autoencoder": (t1 - t0) * 1000,
                "cnn": (t2 - t1) * 1000,
                "ensemble_total": (end - start) * 1000,
            },
        }


def evaluate(root, manifest_path, checkpoint, output):
    manifest = load_manifest(manifest_path, root)
    detector = Detector(checkpoint)
    manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if manifest_hash != detector.metadata["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from the training manifest.")
    records = [r for r in manifest["records"] if r["split"] == "test"]
    with Image.open(Path(root) / records[0]["path"]) as image:
        detector.predict(image)  # warm up, excluded from timing
    rows = []
    for record in records:
        with Image.open(Path(root) / record["path"]) as image:
            prediction = detector.predict(image)
        rows.append(
            {
                "path": record["path"],
                "label": record["label"],
                **prediction["scores"],
                **{f"{k}_ms": v for k, v in prediction["timing_ms"].items()},
            }
        )
    results = {}
    for i, method in enumerate(("autoencoder", "cnn", "ensemble")):
        results[method] = metrics(
            [r["label"] for r in rows],
            [r[method] for r in rows],
            detector.metadata["calibration"]["thresholds"][i],
        )
        key = "ensemble_total_ms" if method == "ensemble" else f"{method}_ms"
        times = [r[key] for r in rows]
        results[method].update(
            {"mean_ms": float(np.mean(times)), "p95_ms": float(np.quantile(times, 0.95))}
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "category": manifest["category"],
        "pretrained": detector.metadata["pretrained"],
        "device": "cpu",
        "warmup_images": 1,
        "metrics": results,
        "timing_scope": "AE/CNN include scoring; ensemble includes resize, both models and fusion; excludes disk I/O.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def save_overlay(image, heatmap, path):
    heatmap = np.asarray(heatmap)
    scaled = (heatmap - heatmap.min()) / max(float(np.ptp(heatmap)), 1e-8)
    color = np.stack([scaled, np.zeros_like(scaled), 1 - scaled], -1)
    overlay = Image.fromarray((color * 255).astype("uint8")).resize(image.size)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.blend(image.convert("RGB"), overlay, 0.4).save(path)
