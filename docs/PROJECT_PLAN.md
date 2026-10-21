# Project plan

## Confirmed scope from the supplied proposal and slides

Group 30, NUS-ISS Pattern Recognition Systems Practice Module. Normal-only visual anomaly detection, an autoencoder, a pretrained CNN method, ensemble comparison, heatmap and simple web demonstration. Dataset undecided. Train/validation/test separation. AUROC, precision, recall and inference time. Deliver source, runnable prototype, trained models, dataset resources, report, slides and demo video. Planned completion: 31 October 2026.

## Draft engineering decisions

Python/PyTorch, Streamlit, MVTec-style adapter, ResNet18 layer2, diagonal feature statistics, 128×128 inputs, top-1% score pooling, 20% normal validation, seed 42, 10 epochs, equal ensemble weights, 95th-percentile thresholds. These defaults are editable through the CLI where exposed and must be recorded with experiments. They were not prescribed by the proposal.

## Suggested remaining schedule

| Dates (2026) | Work | Completion evidence |
| --- | --- | --- |
| 21–22 October | Dataset choice, terms review, category and split audit | Dataset selection record and frozen manifests |
| 23–25 October | Real-data AE and CNN baselines | Checkpoints, training metadata, environment record |
| 26–27 October | Ensemble comparison and error analysis | Three-method metrics and reviewed examples |
| 28–29 October | Demo integration and report | Reproducible demo, report draft, review-log example |
| 30–31 October | Final review, slides and recording | Final report, slides, video and clean reproduction |

This is a suggested schedule, not an assignment of commitments or a guarantee. Owners remain to be agreed by the team. Prefer one fully reproducible category before adding more.

## Backlog

- [ ] Select dataset and categories; document terms and counts.
- [ ] Run real pretrained CNN training and AE training.
- [ ] Record real hold-out results and compare all methods.
- [ ] Inspect false alarms and missed defects.
- [ ] Check robustness to lighting, pose and fine defects with separately designed evaluation.
- [ ] Add pixel-level metrics if the schedule permits.
- [ ] Review demo with a real trained model.
- [ ] Package model weights separately from Git with checksums.
- [ ] Complete report, slides and demo video.
- [ ] Agree repository visibility and software licence before wider sharing.

## Risks

Dataset download or suitability may reduce time for experiments. Normal-only validation cannot optimize recall directly. Ensemble performance can be worse than the strongest baseline. Small sample sizes and unaligned objects can destabilize feature statistics. Retain honest negative results and keep any reduced scope explicit.
