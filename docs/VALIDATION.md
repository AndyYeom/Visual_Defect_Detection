# Release validation

Validated on 30 October 2026.

- Unit/protocol tests: 13 passed, 2 skipped (accelerator-specific tests unavailable on the local CPU host).
- Ruff passed for the original release snapshot; a later notebook update introduced five lint errors, corrected in the 31 October maintenance update below.
- Release ZIP checksum and all model file checksums verified by the installer.
- All 15 exported category models loaded and performed inference on an actual MVTec normal test image on CPU; all returned 224 × 224 maps.
- Bottle normal `test/good/000.png`: original Colab score 27.81625175, exported CPU score 27.81644440; both normal at threshold 31.98647728.
- Bottle defect `test/broken_large/000.png`: original Colab score 61.54000854, exported CPU score 61.53998566; both anomalous at the same threshold.
- Streamlit AppTest: sample selection, inspection, three metric cards, and stale-result clearing after a category change passed.
- Browser inspection confirmed a real defect verdict and original/heatmap display. Screenshot: `docs/demo.png`.

Local environment: Python 3.12, PyTorch 2.15.0, torchvision 0.29.2, anomalib 2.7.0, Streamlit 1.66.1. Original evaluation used Colab Tesla T4; deployment smoke tests used CPU. Cross-device scores are not expected to be bit-identical. The tests do not claim validation on every possible user photograph or operating system.

## Repository maintenance validation — 31 October 2026

- Ruff: all checks passed, including notebooks.
- Unit/protocol tests: 17 passed, 2 skipped on the local CPU host.
- Added regression coverage for a fresh bottle run without legacy results, device propagation, protocol mismatch rejection, overlapping output directories, and fresh/legacy analysis layouts. Training stages are mocked in the fresh-run orchestration test; this is not a repeat of the full GPU experiment.
- All three notebook code-cell sequences passed syntax compilation; Colab execution was not repeated during maintenance.
- Original measured results and released model weights remain unchanged.
- Report and submission slides were checked for English text and metadata. The report's theme font name was normalized to its equivalent English name without changing document content.

GitHub Actions runs the repository checks on Python 3.12, aligned with the quick start. See the repository Actions tab for the status of the latest commit.
