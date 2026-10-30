# System architecture

InspectIQ separates offline experimentation from a lightweight Streamlit inspection app. The final deployed method is PatchCore; the autoencoder, frozen CNN and DINO-based methods remain comparison baselines.

## Offline experiment pipeline

1. `data.py` creates deterministic category manifests: normal training/validation splits and the official test set.
2. `pipeline.py` trains the autoencoder and fits the frozen ResNet-18 feature baseline. `benchmark.py` fits PatchCore and the DINOv2 patch-feature detector and evaluates fixed score ensembles.
3. `suite.py` runs all six methods at 224px across 15 categories. It saves logs, manifests, checkpoints, metrics, predictions and SHA-256 receipts to a persistent result directory. Completed stages are reused only after checksum verification; an interrupted stage restarts.
4. `notebooks/analyze_suite.py` produces the category AUROC comparison, error table and hard-case heatmaps after all 15 categories complete.
5. `scripts/export_patchcore.py` packages evaluated category memory banks and one shared frozen backbone. Export does not retrain models or change thresholds.

The original experiment reused a verified earlier bottle run. Fresh runs now train bottle through the same stage pipeline as every other category. See [reproduction instructions](REPRODUCIBILITY.md).

## Inference application

`app/streamlit_app.py` calls `PatchCoreDetector` in `src/vdd/patchcore_inference.py`. There is no separate API server. The installer and loader verify model checksums; the packaged backbone loads without internet access. CPU and NVIDIA CUDA are supported for the final demo.

The user selects a product category and uploads an image. The backend applies the saved 224px preprocessing, extracts patch features, compares them with that category's normal memory bank, and returns a score and anomaly map. The saved normal-validation threshold determines the verdict. The application displays the original image, heatmap, threshold and inference time.

Inspection state is held in the Streamlit session. Users explicitly download an inspection JSON and overlay PNG; the app does not maintain an append-only review database. Heatmaps are normalized per image for visualization and are not calibrated segmentation masks.

## Boundaries and limitations

Dataset files, runtime results and downloaded weights are excluded from Git. Compact measured results and final submission documents are included. Models are distributed through a versioned GitHub release. The application assumes a known supported product category and representative images; it does not identify arbitrary objects. Human review, operating-threshold validation on new acquisition conditions, and production throughput assessment remain deployment work.
