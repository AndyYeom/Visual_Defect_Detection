import argparse
import json
from pathlib import Path

from PIL import Image

from .data import make_manifest
from .pipeline import Detector, evaluate, save_overlay, train


def main():
    parser = argparse.ArgumentParser(description="Group 30 visual defect detection")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--category", required=True)
    prepare.add_argument("--output", default="artifacts/split.json")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    fit = sub.add_parser("train")
    fit.add_argument("--root", required=True)
    fit.add_argument("--manifest", default="artifacts/split.json")
    fit.add_argument("--output", default="artifacts/baseline")
    fit.add_argument("--epochs", type=int, default=10)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--size", type=int, default=128)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--weight", type=float, default=0.5)
    fit.add_argument("--quantile", type=float, default=0.95)
    fit.add_argument(
        "--offline-smoke",
        action="store_true",
        help="Use RANDOM CNN weights for plumbing tests only, never for reported experiments.",
    )
    test = sub.add_parser("evaluate")
    test.add_argument("--root", required=True)
    test.add_argument("--manifest", default="artifacts/split.json")
    test.add_argument("--checkpoint", default="artifacts/baseline/model.pt")
    test.add_argument("--output", default="reports/baseline")
    predict = sub.add_parser("predict")
    predict.add_argument("--image", required=True)
    predict.add_argument("--checkpoint", default="artifacts/baseline/model.pt")
    predict.add_argument("--output", default="reports/prediction")
    args = parser.parse_args()
    if args.command == "prepare":
        result = make_manifest(
            args.root, args.category, args.output, args.seed, args.validation_fraction
        )
        print(json.dumps({"category": result["category"], "images": len(result["records"])}))
    elif args.command == "train":
        result = train(
            args.root,
            args.manifest,
            args.output,
            args.epochs,
            args.batch_size,
            args.size,
            args.seed,
            args.weight,
            args.quantile,
            not args.offline_smoke,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        print(
            json.dumps(evaluate(args.root, args.manifest, args.checkpoint, args.output), indent=2)
        )
    else:
        detector = Detector(args.checkpoint)
        with Image.open(args.image) as image:
            result = detector.predict(image)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            save_overlay(image, result.pop("heatmap"), output / "overlay.png")
        result["category"] = detector.metadata["category"]
        result["pretrained"] = detector.metadata["pretrained"]
        (output / "prediction.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
