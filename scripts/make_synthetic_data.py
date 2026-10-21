"""Generate tiny synthetic textures for pipeline checks, never benchmark claims."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def generate(root, seed=42):
    rng = np.random.default_rng(seed)
    for split, count in (("train/good", 16), ("test/good", 4), ("test/scratch", 4)):
        folder = Path(root) / "synthetic" / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            x = rng.normal(120, 5, (32, 32, 3)).clip(0, 255).astype("uint8")
            if "scratch" in split:
                x[8:24, 12:16] = 250
            Image.fromarray(x).save(folder / f"{i:03}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/synthetic")
    args = parser.parse_args()
    generate(args.root)
