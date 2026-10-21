# Dataset selection and preparation

The proposal does not select a dataset. First candidate: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). Review the current provider terms, category counts, defects, mask quality and download size before selection. Download manually from the official provider; this repo does not redistribute data.

## Expected layout

```text
data/mvtec_ad/
  bottle/
    train/good/*.png
    test/good/*.png
    test/broken_large/*.png
    test/other_defect/*.png
    ground_truth/broken_large/*_mask.png
```

The loader ignores ground-truth masks for the initial image-level baseline. Non-`good` test folders are labelled anomalous. Each run covers one category. RGB conversion and square resize are applied consistently; no random augmentation is enabled initially. Record any loss of fine detail caused by resizing.

## Split contract

`vdd prepare` takes 20% of `train/good` for normal-only validation with seed 42. Remaining normal images form training. Existing test folders remain held out. All records carry content hashes; duplicates anywhere in the selected category are rejected. Changed images require rebuilding and reviewing the manifest. Manifests use relative paths so the dataset root can move without changing splits.

Exact-byte duplicate checks cannot detect near-duplicates or images of the same item taken in a sequence. Inspect provenance and group related captures before assigning splits for custom datasets. Do not put images from the same physical item into train and test. The current CLI expects a pre-separated MVTec-style layout; custom group splitting needs a reviewed adapter.

The minimum accepted number of normal images is only a software guard, not a scientifically sufficient sample size. Aim for the proposal's few hundred normals where feasible and report actual counts per split and category. A 95th-percentile threshold is unstable on a small validation set.

## Selection record

- Dataset/version and original source:
- Terms reviewed and intended academic use:
- Categories and why they fit the deadline:
- Normal/defect counts by split:
- Annotation coverage and quality:
- Duplicate or related-capture audit:
- Final selection date and reviewer:
