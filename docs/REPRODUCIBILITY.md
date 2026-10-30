# Reproducing the experiments

## Demo versus training

The README quick start runs the released PatchCore models without the dataset or retraining. The commands below fit and evaluate all six methods from scratch. They do not require the team's Google Drive, a source ZIP or a previous bottle experiment.

The original reported experiment is `full15_224_v1`. Keep its results unchanged. Use a new output directory for this revised runner. The scientific settings remain 224px, seed 42, 80/20 normal split, 10 AE epochs, batch size 16, fixed 50/50 fusion and the normal-validation 95th percentile. A new run is not guaranteed to be numerically identical across software versions or devices.

## Environment and source

Use Python 3.12 and a CUDA-enabled PyTorch/torchvision pair for the full experiment. Install PyTorch for your GPU using its official installation instructions, then:

```bash
python -m pip install -e '.[demo,dev]'
python -m ruff check .
python -m pytest -q
```

The project pins anomalib to 2.7.0. The validated local deployment environment is recorded in [VALIDATION.md](VALIDATION.md); it is not a lock file for the original Colab environment. Each experiment writes its actual `environment.txt` and source hashes. Retain these with the source commit and split manifests. First-time training downloads pretrained ResNet, Wide ResNet and DINOv2 weights and requires internet access. The released inference package is self-contained.

## Fresh full comparison

Download and extract MVTec AD separately. Substitute your dataset path below. Run from the repository root:

```bash
python -m vdd.suite --root /path/to/mvtec_anomaly_detection --local artifacts/work/full15_224_reproduction --remote reports/full15_224_reproduction --device cuda
python notebooks/analyze_suite.py --root /path/to/mvtec_anomaly_detection --run reports/full15_224_reproduction
```

`--local` is the working directory; `--remote` is the persistent output directory and may be a local disk or mounted Drive. They must be separate, non-nested directories. Default execution includes all 15 categories. For a smaller first check, use a different run directory and append `--categories bottle`. Analysis/export requires a complete 15-category run. `--device cpu` is available but can be substantially slower; it has not been used to repeat the full benchmark during this repository update.

A category comparison must include all six methods. Completion is recorded in `summary.json`; `status.json` records the latest category and failures. Analysis creates `analysis/image_auroc.png`, `analysis/errors.csv`, `analysis/patchcore_hard_cases.png` and `analysis/anomalydino_hard_cases.png`.

## Resume

Rerun the identical command with the same source checkout and output directories. Verified completed stages are restored; interrupted stages restart rather than resume individual training epochs. Do not change code, categories or settings in an existing run: the protocol guard rejects incompatible resumes. After a runtime reset, restore the same Git commit before running the notebook again.

`--reuse-bottle /path/to/legacy/bottle/run` is an optional compatibility feature for the original flat-layout bottle result. It verifies the recorded configuration; it is not needed for new experiments. Old results generated before this runner revision should be inspected using their archived code, not resumed with changed code.

## Colab

Open `notebooks/Group30_GPU_Full15.ipynb` in Colab, select a GPU runtime, and place only `mvtec_anomaly_detection.tar.xz` in `MyDrive/PRS_Group30`. Run cells in order. The notebook clones GitHub, records the source commit, extracts data locally, and persists results to Drive under a new reproduction run name. On reconnection, keep the run name and source commit unchanged. The Comparison notebook runs the six-method bottle subset; the Baseline notebook runs only AE, CNN and AE + CNN.

## Export evaluated models

After full analysis completes:

```bash
python scripts/export_patchcore.py --run reports/full15_224_reproduction --output artifacts/reproduced_patchcore
```

The exporter packages the evaluated weights and thresholds. A regenerated package has its own checksums and is not interchangeable with the published release configuration. Do not replace `configs/model_release.json` with a new hash without validating and publishing the corresponding model release.
