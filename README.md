# Vehicle Classifier

Classifies cropped traffic-camera vehicle images into 8 classes (`ambulance`, `autobus`, `kamyun`, `kamyunet`, `minibus`, `savari`, `taxi`, `vanet`) with a custom CNN and ResNet18 transfer learning, and flags low-confidence predictions for human review. Built with PyTorch and managed with [uv](https://docs.astral.sh/uv/).

## Documentation

Full project documentation (data audit, experiments, results, analysis) is _online_ on the _Notion_: [link](https://app.notion.com/p/Vehicle-Classifier-Project-Hub-6dc1c6fdc89047e09a4a169cd0b26ad0?source=copy_link)

Available _Offline_ at: `Documentation\Vehicle Classifier — Project Hub 6dc1c6fdc89047e09a4a169cd0b26ad0.html`

## Project structure

```text
.
├── configs/                      # run configurations
├── data/                         # split manifest and normalization stats (generated)
├── dataset/                      # images (not tracked in Git)
├── Figs/                         # figures used in the documentation
├── models/                       # saved models
├── non-relative images for test/ # unrelated (non-vehicle) images for confidence tests
├── notebooks/                    # dataset exploration notebooks
├── outputs/
│   ├── debug/                    # augmentation grid, sample batches
│   ├── final_test/               # one-time evaluation on the frozen test set
│   ├── reports/                  # model comparison tables and plots
│   └── runs/<run_name>/          # best.pt, config.json, history.json, summary.json,
│                                 # val_predictions.npz, analysis/
├── src/vehicle_classifier/       # package source (see below)
├── pyproject.toml                # dependencies (uv)
└── uv.lock
```

### Source modules (`src/vehicle_classifier/`)

| Module                   | Purpose                                                                          |
| ------------------------ | -------------------------------------------------------------------------------- |
| `duplicate_detection.py` | Hash-based duplicate detection across splits                                     |
| `apply_corrections.py`   | Applies the dataset cleaning corrections                                         |
| `splitting.py`           | Stratified train/validation split (manifest)                                     |
| `data.py`                | Manifest loading, letterbox resize, transforms, normalization stats, DataLoaders |
| `model.py`               | Baseline CNN, forward-pass and overfit-one-batch checks                          |
| `resnet18.py`            | ResNet18 variants (frozen, fine-tune, full)                                      |
| `train.py`               | Training loop; best checkpoint selected by validation macro-F1                   |
| `metrics.py`             | Metrics, confusion matrices, review threshold, error analysis                    |
| `analyze.py`             | Per-run analysis written to `outputs/runs/<run_name>/analysis/`                  |
| `compare_runs.py`        | Cross-run comparison written to `outputs/reports/model_comparison/`              |
| `final_test.py`          | Single evaluation on the test set, written to `outputs/final_test/`              |
| `predict.py`             | Predicts one image and returns JSON                                              |
| `utils.py`               | Seeding, console helpers, JSON helpers                                           |

## Setup

```bash
uv sync
```

This creates `.venv` and installs the locked dependencies. Run any module with:

```bash
uv run python -m vehicle_classifier.<module> <args>
```

## Data

The dataset and checkpoints are not stored in Git. Training images are read from `dataset/Combined Dataset/<split>/<class>/`, listed in `data/train_val_split_Manifest.csv`. The other folders in `dataset/` are intermediate cleaning versions. Classes are imbalanced in all splits, and `kamyun` is the smallest class.

## Usage

Typical order:

1. `duplicate_detection` and `apply_corrections`: audit and clean the data
2. `splitting`: create the train/validation manifest
3. `train`: train a run (settings in `configs/`; output in `outputs/runs/<run_name>/`)
4. `analyze` and `compare_runs`: per-run analysis and cross-run comparison
5. `final_test`: evaluate the chosen model once on the test set
6. `predict`: predict a single image

```bash
uv run python -m vehicle_classifier.predict <image_path>
```

Example output:

```json
{
  "predicted_class": "taxi",
  "confidence": 0.87,
  "probabilities": {
    "ambulance": 0.01,
    "autobus": 0.02,
    "kamyun": 0.03,
    "kamyunet": 0.02,
    "minibus": 0.01,
    "savari": 0.02,
    "taxi": 0.87,
    "vanet": 0.02
  },
  "needs_review": false
}
```

`needs_review` is `true` when the confidence is below the threshold chosen on validation data.

## Reproducibility

- The default seed is 42 (Python, NumPy, PyTorch, DataLoader workers).
- Each run saves its full configuration, history, and best checkpoint in `outputs/runs/<run_name>/`.
- The test set is used only once, for the final evaluation.
