# Evaluation and deployment protocol

The final experiment compared six methods on all 15 MVTec AD categories. The selected demo model is PatchCore. Complete aggregate results and category-level results are included in the release model package as `experiment_summary.json` and `category_metrics.csv`.

## Split and calibration

For each category, split official normal training images into 80% training and 20% validation with seed 42. Freeze the split manifest and use the same test images for all methods. Train on normal images only. Set each image decision threshold to the 95th percentile of normal validation scores. Classification uses `score > threshold`. Test labels and masks are used only for reporting metrics, not fitting thresholds or ensemble weights.

All inputs in the final comparison are 224 × 224. Earlier 128px baseline runs are exploratory and are not part of the final table. The completed bottle 224px run was reused with its recorded configuration; the other 14 categories were executed by the resumable suite.

## Methods

- Autoencoder: small convolutional reconstruction model, MSE score, 10 epochs, batch size 16.
- CNN: frozen pretrained ResNet-18 layer2 features with a diagonal Gaussian distance baseline.
- AE + CNN: fixed 50/50 fusion after normal-validation score standardization.
- PatchCore: anomalib 2.7.0, pretrained Wide ResNet-50-2, layer2/layer3, 1% coreset, 9 neighbors.
- DINO: DINOv2 ViT-S/14 patch features and nearest normal feature distances; full memory bank, no foreground mask or rotation augmentation.
- AE + DINO: fixed 50/50 fusion calibrated using normal validation images.

These are practical adaptations. The experiment is not an exact reproduction of published PatchCore or AnomalyDINO benchmark protocols.

## Final results

| Method | Macro image AUROC | Macro F1 |
|---|---:|---:|
| Autoencoder | 0.578116 | 0.271520 |
| CNN | 0.858774 | 0.702750 |
| AE + CNN | 0.819698 | 0.681846 |
| PatchCore | 0.983951 | 0.951429 |
| DINO | 0.967892 | 0.925787 |
| AE + DINO | 0.946434 | 0.909142 |

PatchCore macro pixel AUROC is 0.976323 and pixel AP is 0.542341. Its macro image precision is 0.951016, recall 0.956698, and false-positive rate 0.152242. The official test split contains 1,725 images, including 467 normal and 1,258 anomalous images. Macro metrics average category metrics equally and are not pooled confusion-matrix metrics. AUROC is not classification accuracy.

The high ranking performance does not eliminate false alarms. A threshold derived from normal validation images does not guarantee a 5% test false-positive rate under distribution shift. Do not improve the displayed F1 by tuning thresholds against test labels.

## Deployment package

The exporter reads already evaluated checkpoints, verifies local recovery checkpoints against saved stage SHA-256 receipts, and separates the identical frozen backbone from category-specific memory banks. It checks backbone tensor equality across all categories. Thresholds and preprocessing metadata are preserved. No training or recalibration happens during export or in the app.

The installer verifies the entire release ZIP; the inference loader checks the selected model and backbone hashes. Loading uses PyTorch `weights_only=True`, and the pretrained backbone constructor has downloads disabled. CPU and CUDA are supported. GPU availability is checked before CUDA selection.

## Heatmaps and latency

Anomaly maps are resized and overlaid on the input for human review. The display normalizes each map independently, so color intensity cannot be compared quantitatively across different images. The overlay is not a calibrated binary defect mask. Pixel AUROC/AP are reported from experimental maps; PRO is not implemented.

Displayed inference time excludes loading a model, checksum verification, disk I/O, and rendering. The first inspection can therefore take longer than the displayed milliseconds. Performance depends on hardware and is not a production throughput guarantee.

## Reproducibility limits

The experiment ran on a Colab Tesla T4. Different PyTorch versions, CPU/GPU kernels, or hardware may cause small floating-point differences. Borderline scores near the threshold require review. Report the original protocol and software versions alongside comparisons; do not claim production reliability, universal category recognition, or published SOTA reproduction.
