<!--
File: README.md
Project: Intelligent Fruit Classification and Information System
Description: Explains which trained model artefacts are required,
how to obtain them, their expected location, and any relevant
versioning or loading instructions.
Author: Shubhan Singh
Responsible Area: Shared/Core
Maintainers: Group 8
-->
# Model Artefacts

This folder holds the trained model and the supporting files that the application loads at start-up. The application **never retrains the model**: it loads the files below once, and uses them to classify uploaded images. Model development and evaluation were carried out offline in the project notebooks (see [`docs/model_card.md`](../docs/model_card.md)).

## 1. Required files

All three files must be present in this folder, and must come from the **same model run**.

| File | Size | Purpose | Produced by |
|------|------|---------|-------------|
| `M4_PC1.pt` | 15.6 MiB | Trained weights (a PyTorch `state_dict`) of the selected model: EfficientNetB0, fine-tuned, run `M4_PC1`. | Training notebook `Model_Training_and_Experiments_PC1.ipynb`, saved as `results/checkpoints/M4_PC1.pt`. |
| `final_model_selection.json` | 1.5 KiB | Deployment record: selected model, checkpoint run, configuration hash, dataset fingerprint, preprocessing version and hyperparameters. | `Results_Collation_and_Comparison.ipynb`, Section 19 (Deployment Record). |
| `m4_ood_stats.npz` | 5.8 MiB | Class centroids and inverse covariance of the model's feature space. The application uses them to measure how far an uploaded image lies from the known fruit images (the *feature-distance* signal). | `src/scripts/export_ood_stats.py` (Section 3 below). |

The three files together are about 22 MiB, which is within the limits of ordinary Git hosting (GitHub rejects files above 100 MiB), so they can be committed with the repository.

## 2. What the ApplicationVerifies when it Starts

`src/ml/model_service.py` refuses to start, and the application shows a clear error instead of making predictions, if any of the following fails:

1. One of the three files is missing.
2. `preprocessing_version` in `final_model_selection.json` differs from `PREPROCESSING_VERSION` in `src/fruit_classifier/config.py` (the model was trained on differently prepared images).
3. The checkpoint file name differs from the `checkpoint_path` recorded in the JSON file (a different model was supplied).
4. The weights do not match the network architecture exactly (the file is corrupt or belongs to another architecture).
5. `m4_ood_stats.npz` does not contain a `(2, 1280)` centroid array and a `(1280, 1280)` inverse covariance matrix.

The model version recorded with every prediction in the history table is `<checkpoint run>-<configuration hash>`, currently `M4_PC1-d7e2e21cf167`.

## 3. Generating `m4_ood_stats.npz`

The statistics are rebuilt from two files that the training notebook already saved, so no images or dataset are needed.

| Input | Location in the notebook output | Required |
|-------|---------------------------------|----------|
| `M4_PC1_embeddings.npz` | `results/embeddings/` | Yes |
| `ood_examples_results_PC1.csv` | `results/ood/` | No (adds the false-acceptance columns) |
| `M4_PC1.pt` | `results/checkpoints/` | No (adds the Accepted / Needs review / Rejected report) |

Copy the inputs into this folder (they are not committed; see Section 6), then run from the project root:

```powershell
python src/scripts/export_ood_stats.py --embeddings artifacts/M4_PC1_embeddings.npz --ood-csv artifacts/ood_examples_results_PC1.csv --checkpoint artifacts/M4_PC1.pt
```

The script writes `artifacts/m4_ood_stats.npz` and prints three things:

1. **A parity check.** It recomputes the 95th percentile of the training distances and compares it with the value printed by the training notebook (35.7256 for `M4_PC1`). The message `MATCHES the notebook` confirms that the statistics were rebuilt correctly. If it reports `DIFFERS`, the embeddings file does not belong to this checkpoint.
2. **A threshold table.** For candidate distance limits, it shows the share of genuine test images kept and the share of unsupported images wrongly accepted. This is the evidence for choosing `DISTANCE_REJECT_THRESHOLD` in `src/fruit_classifier/config.py`.
3. **An outcome report** (with `--checkpoint`). For the thresholds in `config.py` and a grid of alternatives, it shows how genuine and unsupported images would be split into Accepted, Needs review and Rejected.

The distance limit and the confidence threshold are **properties of one specific checkpoint**. Whenever the checkpoint changes, regenerate the statistics and review the thresholds.

## 4. Which Checkpoint is Deployed

Four independent training runs of the selected model exist (`M4_PC1` to `M4_PC4`, trained on four computers with identical code, data and configuration). On the held-out test set all four score 382 of 382. Their weights are not byte-identical: `M4_PC2`, `M4_PC3` and `M4_PC4` agree with one another to floating-point precision, while `M4_PC1` (trained with PyTorch 2.7.1 instead of 2.14.1) differs from them by about 0.09% in relative weight distance, confined to the layers that were fine-tuned. The collation notebook selected `M4_PC1` as the first of four runs tied on accuracy. [`docs/model_card.md`](../docs/model_card.md), Section 8, documents the selection process and the comparison in full.

**Consistency rule.** The weights, `m4_ood_stats.npz`, the distance and confidence thresholds, and `final_model_selection.json` must all come from the same run. Feature vectors differ by about 1% between `M4_PC1` and the other runs, so statistics and thresholds calibrated for one run are not valid for another.

## 5. Troubleshooting

| Message shown by the application | Cause and remedy |
|----------------------------------|------------------|
| `Required model file not found: ...` | One of the three files is missing from this folder. Add it (Section 1). |
| `Preprocessing mismatch: ...` | The model and the application disagree on how images are prepared. Use a matching `final_model_selection.json`, or update `PREPROCESSING_VERSION` only if the application's preprocessing really matches. |
| `final_model_selection.json names ... but ... was supplied` | The JSON file and the checkpoint belong to different runs. |
| `Could not load model weights ...` | The checkpoint is corrupt, truncated or from a different architecture. Copy it again from the notebook output. |
| `... has unexpected shapes ...` | `m4_ood_stats.npz` was not built for this model. Regenerate it (Section 3). |
