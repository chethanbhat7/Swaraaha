# Evaluation

Standalone CLI for benchmarking trained Swaraaha models against the validation
split. Produces console reports plus JSON/PNG artifacts under `reports/` (or any
`--output_dir` you choose).

> This is a manual tool. Nothing imports `model.evaluation` at runtime — run it
> directly whenever you want to check how a trained checkpoint performs.

## When to use it

- After training a classifier or localizer, to get precision/recall/F1, AUROC,
  AUPRC, specificity, and a confusion matrix on the held-out 20% val split.
- To find a better decision threshold without retraining
  (`--sweep_thresholds`).
- To inspect which samples a model gets wrong (`--save_misclassified`).

## Evaluate a classifier

```bash
python -m model.evaluation.evaluate \
    --model_type classifier \
    --class_name prolongation \
    --model_path model/weights/prolongation_<fingerprint>_best.pt \
    --data_dir data
```

Produces:
- `reports/prolongation_report.json` — full metrics (incl. binary/AUROC/AUPRC)
- `reports/prolongation_confusion_matrix.png`
- `reports/prolongation_misclassified.json` (with `--save_misclassified`)

### Threshold sweep

```bash
python -m model.evaluation.evaluate \
    --model_type classifier --class_name block \
    --model_path model/weights/block_<fingerprint>_best.pt \
    --data_dir data \
    --sweep_thresholds --save_misclassified
```

Sweeps t ∈ [0.1, 0.9] and reports the thresholds maximizing F1, specificity,
and Youden's J in the saved JSON (`metrics["threshold_sweep"]`).

## Evaluate the CNN localizer

```bash
python -m model.evaluation.evaluate \
    --model_type localizer \
    --model_path model/weights/<localizer_fingerprint>_best.pt \
    --data_dir data \
    --n_mels 128 --hop_length 512
```

Produces `reports/localizer_report.json` with frame-level and event-level
localization metrics.

## All flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--model_type` | — | `classifier` or `localizer` (required) |
| `--class_name` | — | Dysfluency class; required for `classifier` |
| `--model_path` | — | Trained checkpoint to evaluate (required) |
| `--data_dir` | `data` | Root data directory |
| `--output_dir` | `model/evaluation/reports` | Where reports/plots are saved |
| `--batch_size` | `8` | Eval batch size |
| `--max_length_seconds` | `10.0` | Max audio length |
| `--threshold` | `0.5` | Decision threshold |
| `--save_misclassified` | off | Write misclassified sample paths to JSON |
| `--sweep_thresholds` | off | Run threshold sweep + optimal-threshold search |
| `--n_mels` | `128` | Mel bins (localizer only) |
| `--hop_length` | `512` | Spectrogram hop length (localizer only) |

## Metrics module (`model/evaluation/metrics.py`)

Programmatic metric functions used by the CLI (also importable for custom
experiments):

- `compute_classification_metrics` — accuracy, macro/weighted P/R/F1, support
- `compute_binary_metrics` — P/R/F1, specificity, AUROC, AUPRC
- `compute_multilabel_metrics` — subset accuracy, hamming loss, per-class
- `compute_localization_metrics` — frame + event-level localization metrics
- `find_optimal_threshold` — threshold maximizing a given metric
- `confusion_matrix` / `save_confusion_matrix_plot` — matrix + PNG
- `save_report` / `print_*_report` — JSON + console output
