"""Pinned Anomalib models evaluated with the project's normal-only split.

This is an adapted full-shot benchmark, not a reproduction of paper scores.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from .data import ImageDataset, load_manifest
from .device import device_name, resolve_device, synchronize
from .pipeline import Detector, save_overlay
from .scoring import calibrate, combine, metrics

SETTINGS = {
    "patchcore": {"backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"],
                  "num_neighbors": 9, "coreset_ratio": 0.01},
    "anomalydino": {"encoder_name": "vit_small_patch14_dinov2", "num_neighbours": 1,
                    "masking": False, "coreset_subsampling": False},
}


def build_model(method):
    if method == "patchcore":
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        return PatchcoreModel(layers=SETTINGS[method]["layers"],
                              backbone=SETTINGS[method]["backbone"], num_neighbors=9)
    from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel
    return AnomalyDINOModel(**SETTINGS[method])


def preprocessing(size, method):
    interpolation = InterpolationMode.BICUBIC if method == "anomalydino" else InterpolationMode.BILINEAR
    return Compose([Resize((size, size), interpolation=interpolation, antialias=True),
                    ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def calibration(scores, quantile=0.95):
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration requires finite normal validation scores")
    return {"threshold": float(np.quantile(scores, quantile)), "quantile": quantile,
            "mean": float(scores.mean()), "std": float(max(scores.std(), 1e-8)), "n": len(scores)}


def mask_for(root, record, size):
    if record["label"] == 0:
        return np.zeros((size, size), dtype=np.uint8)
    p = Path(record["path"])
    mask = Path(root) / p.parts[0] / "ground_truth" / p.parts[-2] / (p.stem + "_mask.png")
    with Image.open(mask) as image:
        return (np.asarray(image.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def read_records(root, records, size, method, batch_size=8):
    dataset = ImageDataset(root, records, size)
    dataset.transform = preprocessing(size, method)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def fit_memory(model, loader, method):
    model.train()
    model.requires_grad_(False)
    # Keep pretrained BN/dropout frozen while the outer module collects embeddings.
    extractor = model.feature_extractor if method == "patchcore" else model.feature_encoder
    extractor.eval()
    device = next(model.parameters()).device
    for i, (x, _, _) in enumerate(loader):
        model(x.to(device))
        if i % 5 == 0:
            print(f"Embedding batches {i + 1}/{len(loader)}", flush=True)
    print("Finalizing normal memory bank", flush=True)
    if method == "patchcore":
        model.subsample_embedding(sampling_ratio=SETTINGS[method]["coreset_ratio"])
    else:
        model.fit()
    model.eval()


@torch.no_grad()
def predict_one(model, image, transform, device):
    synchronize(device)
    start = time.perf_counter()
    x = transform(image.convert("RGB"))[None].to(device)
    output = model(x)
    score = float(output.pred_score.flatten()[0].cpu())
    heatmap = output.anomaly_map.squeeze().cpu().numpy()
    synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000
    if not np.isfinite(score) or not np.isfinite(heatmap).all():
        raise ValueError("Nonfinite prediction")
    return score, heatmap, elapsed


def write_rows(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(root, manifest_path, output, method, size=224, device="cuda", seed=42):
    if size < 32 or size % 8 or size % 14:
        raise ValueError("Use size divisible by both 8 and 14, e.g. 224 or 448")
    if importlib.metadata.version("anomalib") != "2.7.0":
        raise ValueError("This adapter is validated against anomalib==2.7.0")
    device = resolve_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    manifest = load_manifest(manifest_path, root)
    groups = {s: [r for r in manifest["records"] if r["split"] == s] for s in ["train", "val", "test"]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "method": method, "implementation": "anomalib==2.7.0 torch model adapter",
        "category": manifest["category"], "size": size, "seed": seed,
        "settings": SETTINGS[method], "split_counts": {k: len(v) for k, v in groups.items()},
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "device": str(device), "device_name": device_name(device),
        "protocol": "Full-shot training split only; no rotations/masking; resize without crop; normal-only 95% calibration",
        "pixel_protocol": "All test images at resized evaluation resolution; nearest-neighbor ground truth resizing",
        "timing_scope": "RGB conversion, resize, normalization, device transfers and inference; excludes disk I/O and model loading",
        "versions": {n: importlib.metadata.version(n) for n in ["anomalib", "timm", "torch", "torchvision"]},
    }
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    model = build_model(method).to(device)
    transform = preprocessing(size, method)
    synchronize(device)
    start = time.perf_counter()
    fit_memory(model, read_records(root, groups["train"], size, method), method)
    synchronize(device)
    metadata["fit_seconds"] = time.perf_counter() - start
    metadata["memory_bank_shape"] = list(model.memory_bank.shape)
    print("Calibrating on held-out normal images", flush=True)
    val_rows = []
    for record in groups["val"]:
        with Image.open(Path(root) / record["path"]) as image:
            score, _, _ = predict_one(model, image, transform, device)
        val_rows.append({"path": record["path"], "label": 0, "score": score})
    cal = calibration([r["score"] for r in val_rows])
    metadata["calibration"] = cal
    write_rows(output / "validation.csv", val_rows)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": metadata}, output / "model.pt")
    # Verify dynamic memory bank and weights survive loading into the same architecture.
    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True)["state_dict"])
    (output / "training.json").write_text(json.dumps(metadata, indent=2))
    with Image.open(Path(root) / groups["val"][0]["path"]) as image:
        for _ in range(3):
            predict_one(model, image, transform, device)
    print("Evaluating untouched test manifest", flush=True)
    rows, maps, masks = [], [], []
    for i, record in enumerate(groups["test"]):
        with Image.open(Path(root) / record["path"]) as image:
            score, heatmap, ms = predict_one(model, image, transform, device)
            if i < 3:
                save_overlay(image, heatmap, output / "examples" / f"{i:03}_overlay.png")
        rows.append({"path": record["path"], "label": record["label"], "score": score,
                     "prediction": int(score > cal["threshold"]), "total_ms": ms})
        maps.append(heatmap.astype(np.float32))
        masks.append(mask_for(root, record, size))
    write_rows(output / "predictions.csv", rows)
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    result = metrics(labels, scores, cal["threshold"])
    y, s = np.asarray(masks).ravel(), np.asarray(maps).ravel()
    result.update({"pixel_auroc": float(roc_auc_score(y, s)),
                   "pixel_ap": float(average_precision_score(y, s)),
                   "mean_ms": float(np.mean([r["total_ms"] for r in rows])),
                   "p95_ms": float(np.quantile([r["total_ms"] for r in rows], .95)),
                   "false_positives": sum(r["prediction"] == 1 and r["label"] == 0 for r in rows),
                   "false_negatives": sum(r["prediction"] == 0 and r["label"] == 1 for r in rows)})
    np.savez_compressed(output / "anomaly_maps.npz", paths=np.array([r["path"] for r in rows]), maps=np.array(maps))
    (output / "metrics.json").write_text(json.dumps({**metadata, "metrics": result}, indent=2, allow_nan=False))
    print(json.dumps({"method": method, "metrics": result}), flush=True)
    return result


def evaluate_ae_dino(root, manifest_path, checkpoint, dino_dir, output, device="cuda"):
    """Fixed 50/50 ensemble. Align by exact paths; calibrate only normal val."""
    manifest = load_manifest(manifest_path, root)
    detector = Detector(checkpoint, device=device)
    manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    dino = json.loads((Path(dino_dir) / "training.json").read_text())
    if any(m["manifest_sha256"] != manifest_hash for m in [detector.metadata, dino]):
        raise ValueError("Baseline/DINO split mismatch")
    if detector.metadata["size"] != dino["size"]:
        raise ValueError("Use matched image resolutions")
    splits = {}
    for split, name in [("val", "validation.csv"), ("test", "predictions.csv")]:
        with (Path(dino_dir) / name).open() as f:
            dino_rows = {r["path"]: r for r in csv.DictReader(f)}
        records = [r for r in manifest["records"] if r["split"] == split]
        if set(dino_rows) != {r["path"] for r in records}:
            raise ValueError("Missing, extra or misaligned DINO predictions")
        scores = []
        for r in records:
            with Image.open(Path(root) / r["path"]) as image:
                prediction = detector.predict(image)
            scores.append([prediction["scores"]["autoencoder"], float(dino_rows[r["path"]]["score"])])
        splits[split] = (records, scores)
    cal = calibrate(splits["val"][1], weight=.5, quantile=.95)
    values = combine(splits["test"][1], cal)[:, 2]
    labels = [r["label"] for r in splits["test"][0]]
    result = metrics(labels, values, cal["thresholds"][2])
    result["false_positives"] = int(sum((values > cal["thresholds"][2]) & (np.asarray(labels) == 0)))
    result["false_negatives"] = int(sum((values <= cal["thresholds"][2]) & (np.asarray(labels) == 1)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps({"method": "ae_anomalydino", "manifest_sha256": manifest_hash,
        "calibration": cal, "metrics": result, "note": "Image-score ensemble only; no pixel metric or latency claim"}, indent=2))
    write_rows(output / "predictions.csv", [{"path": r["path"], "label": r["label"], "score": float(s)}
               for r, s in zip(splits["test"][0], values, strict=True)])
    print(json.dumps({"method": "ae_anomalydino", "metrics": result}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["patchcore", "anomalydino", "ae_anomalydino"], required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint")
    p.add_argument("--dino-dir")
    args = p.parse_args()
    if args.method == "ae_anomalydino":
        evaluate_ae_dino(args.root, args.manifest, args.checkpoint, args.dino_dir, args.output, args.device)
    else:
        run_benchmark(args.root, args.manifest, args.output, args.method, args.size, args.device)


if __name__ == "__main__":
    main()
