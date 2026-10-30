# Dataset: MVTec AD

The final project uses the original **MVTec AD** dataset, not MVTec AD 2. Obtain it from the [official provider](https://www.mvtec.com/company/research/datasets/mvtec-ad) and follow the provider's dataset terms. Images are not redistributed in this repository.

## Coverage

The experiment covers all 15 categories: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood and zipper. These include objects and textures, with normal training images and normal/defective test images. Defective test images have pixel-level ground-truth masks.

The dataset contains 3,629 official training images and 1,725 test images (467 normal and 1,258 anomalous). Our seed-42 category-wise split uses 2,903 normal images for fitting and 726 for validation. Test images and masks are used only for evaluation, never threshold fitting or model selection within a run.

## Directory layout

```text
mvtec_anomaly_detection/
  bottle/
    train/good/*.png
    test/good/*.png
    test/broken_large/*.png
    ground_truth/broken_large/*_mask.png
  cable/
  ...
```

Pass the directory containing the 15 category folders as `--root`. Keep original folder names. Images are resized to 224 x 224 for the final comparison. Use a local working disk for training and a separate persistent folder for experiment outputs. No dataset is needed to inspect an uploaded image with the released demo models.

## Evaluation integrity

Split manifests preserve image membership across all six methods. Validation uses only normal samples; the 95th percentile sets the image threshold. Defect types, test labels and masks support image metrics, pixel metrics and post-evaluation error analysis. Error analysis describes failure modes; it does not retroactively tune the reported experiment. See [evaluation protocol](EVALUATION.md).
