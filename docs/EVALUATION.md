# Evaluation protocol

## Freeze before opening the test results

Use normal training images to optimize the autoencoder and estimate CNN means/variances. Use normal validation images only to standardize both image scores and determine each decision threshold. Fix the ensemble weight at 0.5 before evaluation. Do not optimize recall, AUROC or weights against the held-out defect images.

This first implementation uses a fixed epoch count. If adding early stopping or extensive model selection, reserve a separate normal calibration split; repeatedly selecting models on the same calibration set can overfit it. Do not use test labels to choose the dataset category or report only a favourable run.

## Metrics

Report autoencoder, CNN and ensemble on the identical test manifest: image AUROC, precision, recall, F1 and false-positive rate. AUROC is null when only one class exists; it must not be reported as zero. Thresholded metrics use `score > threshold`. The confusion-matrix counts and false-alarm examples can be derived from predictions.csv and the saved thresholds.

Record category, split counts/hash, image size, seed, epoch count, package versions and checkpoint. Compare per category; do not pool differently calibrated categories without a justified protocol. The baseline is not guaranteed to improve through ensembling.

Latency is CPU-only in this scaffold. One test image warms the model, then each image is scored once. AE/CNN component times include scoring; ensemble_total includes RGB conversion, resize, both models and fusion, excludes disk I/O and model loading. Report hardware, thread count (2), mean and p95, and distinguish these measurements from throughput or production latency. Repeat timing measurements for final reporting.

## Heatmap review

Inspect representative normal images, detected anomalies, false positives and false negatives. Relative overlay colours cannot be compared quantitatively across images. The fused map uses image-score calibration for visualization and is not a learned segmentation mask. Pixel AUROC/PRO, mask alignment and localization thresholds remain unimplemented extensions.

## Review feedback

Demo overrides are recorded with image hash, checkpoint identifier, scores and threshold. Do not silently fold test-set reviews into training or threshold selection. Reserve later development data and a new untouched test set for subsequent improvements.

## Results table (unmeasured)

| Category | Method | AUROC | Precision | Recall | FPR | Mean ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | Autoencoder | — | — | — | — | — | — |
| TBD | CNN | — | — | — | — | — | — |
| TBD | Ensemble | — | — | — | — | — | — |

Synthetic smoke-test results must never replace the blank benchmark entries.
