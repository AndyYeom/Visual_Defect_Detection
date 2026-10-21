"""MVTec-style images and deterministic, content-disjoint split manifests."""

import hashlib
import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTENSIONS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(root, category, output, seed=42, validation_fraction=0.2):
    root = Path(root).resolve()
    category_path = root / category
    normal = images(category_path / "train" / "good")
    test = images(category_path / "test")
    if len(normal) < 5 or not test:
        raise ValueError("Need >=5 train/good images and a non-empty test directory.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    if any(p.parent.name != "good" for p in images(category_path / "train")):
        raise ValueError("Training must contain normal images only (train/good).")
    hashes = [digest(p) for p in normal + test]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate image content detected. Deduplicate before splitting.")
    random.Random(seed).shuffle(normal)
    n_val = max(1, min(len(normal) - 1, round(len(normal) * validation_fraction)))
    records = []
    for split, paths in (("val", normal[:n_val]), ("train", normal[n_val:]), ("test", test)):
        for p in paths:
            records.append(
                {
                    "path": str(p.relative_to(root)),
                    "split": split,
                    "label": int(p.parent.name != "good"),
                    "sha256": digest(p),
                }
            )
    result = {"schema_version": 1, "category": category, "seed": seed, "records": records}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def load_manifest(path, root):
    manifest = json.loads(Path(path).read_text())
    root = Path(root).resolve()
    seen = set()
    for record in manifest["records"]:
        p = (root / record["path"]).resolve()
        if not p.is_relative_to(root):
            raise ValueError("Image path escapes dataset root.")
        if record["split"] not in {"train", "val", "test"} or record["label"] not in {0, 1}:
            raise ValueError("Invalid split or label.")
        if record["split"] != "test" and record["label"] != 0:
            raise ValueError("Train and validation must be normal-only.")
        actual = digest(p)
        if actual != record["sha256"] or actual in seen:
            raise ValueError("Changed or duplicate image. Rebuild and review the manifest.")
        seen.add(actual)
    if {r["split"] for r in manifest["records"]} != {"train", "val", "test"}:
        raise ValueError("All three splits are required.")
    return manifest


def transform(size):
    if size < 32 or size % 8:
        raise ValueError("Image size must be >=32 and divisible by 8.")
    return Compose([Resize((size, size)), ToTensor()])


class ImageDataset(Dataset):
    def __init__(self, root, records, size):
        self.root, self.records, self.transform = Path(root), records, transform(size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record["label"], record["path"]
