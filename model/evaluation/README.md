# Evaluation

CLI tools for evaluating trained Swaraaha models and generating JSON/plot/report artifacts.

> `model.evaluation` is primarily used as a manual/offline evaluation tool. Reports and plots are written under `model/evaluation/reports/` by default unless you set `--output_dir`.

## Main CLI: `model.evaluation.evaluate`

```bash
python -m model.evaluation.evaluate --help
```

Supported modes (`--model_type`):
- `classifier` — evaluate a single dysfluency classifier (requires `--class_name`; use `--model_path` or `--all` with `--registry`).
- `localizer` — evaluate a localizer checkpoint (`--localizer_type cnn|wav2vec2`).
- `multitask` — evaluate a multitask classifier.

### Common options
- `--data_dir` (default: `data`)
- `--output_dir` (default: `model/evaluation/reports`)
- `--registry` (default: `model/registry.json`)
- `--batch_size` (default: `8`)
- `--max_length_seconds` (default: `10.0`)
- `--threshold` (default: `0.5`)
- `--full` — evaluate the entire `--data_dir` (no 20% split). Useful with prepared splits like `data/test`.
- `--save_misclassified`
- `--sweep_thresholds`

### Classifier examples

```bash
# Single classifier on a prepared test split
python -m model.evaluation.evaluate \
  --model_type classifier \
  --class_name prolongation \
  --model_path model/weights/<prolongation_checkpoint>_best.pt \
  --data_dir data/test --full

# Evaluate all five single classifiers using paths from registry.json
python -m model.evaluation.evaluate \
  --model_type classifier --all \
  --data_dir data/test --full

# Threshold sweep / save misclassified samples
python -m model.evaluation.evaluate \
  --model_type classifier --class_name block \
  --model_path model/weights/<block_checkpoint>_best.pt \
  --data_dir data --sweep_thresholds --save_misclassified
```

### Localizer examples

```bash
# CNN localizer
python -m model.evaluation.evaluate \
  --model_type localizer --localizer_type cnn \
  --model_path model/weights/<cnn_localizer>_best.pt \
  --data_dir data/test --full \
  --n_mels 128 --n_fft 1024 --hop_length 512

# Wav2Vec2 localizer
python -m model.evaluation.evaluate \
  --model_type localizer --localizer_type wav2vec2 \
  --model_path model/weights/<w2v2_localizer>_best.pt \
  --data_dir data/test --full
```

### Multitask example

```bash
python -m model.evaluation.evaluate \
  --model_type multitask \
  --model_path model/weights/<multitask_checkpoint>_best.pt \
  --data_dir data/test --full \
  --thresholds_path path/to/multitask_thresholds.json
```

Additional options:
- `--thresholds_path` — apply per-class thresholds at evaluation time (multitask workflows)
- `--sources` — comma-separated dataset source filter (e.g. `boli`) when label metadata includes `sources.csv`

## Other evaluation utilities

- `model.evaluation.full_evaluate` — broader evaluation/report generation workflows.
- `model.evaluation.presentation` — generates presentation charts/markdown (e.g. `model/evaluation/reports/presentation/`).
- `model.evaluation.probe_combiner` — probes localizer/saliency mismatch behavior.

```bash
python -m model.evaluation.probe_combiner --data_dir data/test --max_length_seconds 3
```

## Outputs and report policy

- Evaluation runs may generate JSON, plots, and other artifacts under `--output_dir`.
- This repository tracks a **curated subset** of reports/snapshots (e.g. `SUMMARY.md`, `comparative_study_report.md/.json`, `presentation/presentation.md` and presentation PNGs) and ignores regenerable sweep/arm outputs via `model/evaluation/reports/.gitignore`.
- Treat curated reports as **snapshots**: regenerate intentionally and commit deliberately.

## Metrics module

Programmatic metrics and helpers are in `model/evaluation/metrics.py`. Use the CLI `--help` and source code as the definitive reference for flags and behavior.
