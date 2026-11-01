# HDBD takeover: controlled A–D comparison

The target remains the existing binary `future_keydown` proxy: a `main_keydown` at the window's final timestep or within the next 10 samples gives label 1. The index uses 30 samples at 10 Hz, stride 1, and does not cross CSV session boundaries. Participant split seeds 0–4 each partition 28 people into 20 train, 4 validation, and 4 test. The score is window-level ROC-AUC. Choose the best epoch using validation ROC-AUC, then evaluate the held-out test partition once. No threshold is involved in ROC-AUC.

## Architecture and preprocessing

| Setting | Model | CAN preprocessing |
| --- | --- | --- |
| A | Existing 3D-CNN + signals MLP | Legacy global min–max and clipping |
| B | Existing 3D-CNN + signals MLP | Train-only z-score |
| C | Patch/frame Transformer + time-token Transformer | Legacy global min–max and clipping |
| D | Patch/frame Transformer + time-token Transformer | Train-only z-score |

The visual Transformer applies a 16×16 convolutional patch embedding to each paired scene/gaze frame, pools the patch grid to 2×2 spatial positions, and processes 30×4=120 tokens with learned time and spatial positions. It has dimension 128, 4 heads, 2 layers, and dropout 0.1. The signal Transformer projects each of 30 six-feature timesteps to dimension 64, adds learned time positions, and uses 4 heads, 2 layers, and dropout 0.1. Both mean-pool tokens. The 9D one-hot HMI context from the last timestep remains categorical metadata and joins the two embeddings at the final fusion. All windows are fixed length; no padding mask is needed.

For z-score, each of `Throttle`, `RPM`, `Steering`, and `Speed` is fit on unique timesteps covered by the selected training windows of that split. Missing/nonfinite values are omitted from fitting and become zero after normalization. A feature with no valid observations uses mean 0 and scale 1. A feature with standard deviation below 1e-6 uses scale 1, so a constant observed value becomes zero. The train statistics are stored in every checkpoint and reused for validation, test, and standalone evaluation. Physiological normalization is unchanged from the baseline: participant-level statistics are computed over that participant's available CSV sessions. This is a separate protocol limitation for later study; it is the same in all four settings.

## Full training commands

Run from the repository root with the full dataset and index available. All four commands inherit the same five splits, seed, batch size, epochs, learning rate, and checkpoint rule from `configs/paper_full_reproduction.json`.

```powershell
python scripts/train_paper_baseline.py --config configs/paper_full_reproduction.json --architecture baseline --can-normalization legacy --run-name A-baseline-legacy
python scripts/train_paper_baseline.py --config configs/paper_full_reproduction.json --architecture baseline --can-normalization zscore --run-name B-baseline-zscore
python scripts/train_paper_baseline.py --config configs/paper_full_reproduction.json --architecture transformer --can-normalization legacy --run-name C-transformer-legacy
python scripts/train_paper_baseline.py --config configs/paper_full_reproduction.json --architecture transformer --can-normalization zscore --run-name D-transformer-zscore
```

Use the same `--bundle`, `--index`, and `--cache-dir` overrides on every command if your paths differ from the config. Each run writes `summary.json`, `history.jsonl`, and `checkpoints/split_seed_{0..4}_best.pt` under its printed `run_dir`. The summary reports AUC per split, mean, sample standard deviation, and best single test run. The model is selected only by validation AUC.

## Standalone evaluation

Use this command for **each** best checkpoint from A, B, C, and D, substituting its path and the required local dataset path. The checkpoint supplies the split seed, architecture, preprocessing statistics, and heatmap variant.

```powershell
python scripts/evaluate_takeover_checkpoint.py --checkpoint experiments/<run_id>/checkpoints/split_seed_0_best.pt --bundle <path-to-hdbd.tar.gz> --index data/interim/paper_window_index.csv --device cuda
```

Repeat with split seeds 1–4. The full training command already performs test evaluation after loading each best checkpoint; standalone evaluation is for reproduction or verification.

## Thesis results table

| Setting | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ROC-AUC | SD | Best single run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: original baseline, legacy CAN (recorded DGX run) | 0.819056 | 0.905818 | 0.823374 | 0.806090 | 0.821112 | 0.835090 | 0.040103 | 0.905818 |
| B: baseline, CAN z-score | — | — | — | — | — | — | — | — |
| C: Transformer, legacy CAN | — | — | — | — | — | — | — | — |
| D: Transformer, CAN z-score | — | — | — | — | — | — | — | — |

The A row is the existing `experiments/20260423_210328_dgx-full-5split/summary.json`, not a new run. Its SD is the sample standard deviation across five split results. The paper's reported 0.8615 detection AUC is context, not a direct matched-protocol comparison: this repository uses a `main_keydown` proxy and a 10-step future label rule. Report any paper figure with this distinction.

## Verification

```powershell
python -m pytest -q tests/test_transformer_can_pipeline.py tests/test_train_paper_baseline_helpers.py tests/test_hdbd_paper_dataset_session_sequence.py
```

The test suite checks train-only CAN fitting using an excluded extreme row, missing/constant values, Transformer tensor shapes, checkpoint metadata and reload, and one backward/optimizer step. It does not establish performance on the real HDBD archive.
