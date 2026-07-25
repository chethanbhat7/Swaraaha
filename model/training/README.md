# Training Pipelines

---

## Prerequisites

```bash
pip install -r model/requirements.txt
```

You also need a prepared dataset. See [`model/data/README.md`](../data/README.md) for setup.

---

## Overview

Three training pipelines, one per model type:

| Pipeline | Script | Model | Input | Output |
|----------|--------|-------|-------|--------|
| Classifier | `train_classifier.py` | Wav2Vec 2.0 binary classifiers | Raw audio | `{class}_best.pt` |
| CNN Localizer | `train_localizer.py` | CNN over spectrograms | Mel-spectrograms | `localizer_best.pt` |
| Wav2Vec2 Localizer | `train_wav2vec2_localizer.py` | Wav2Vec2 + temporal attention | Raw audio | `w2v2_localizer_best.pt` |

All weights save to `model/weights/` by default.

---

## Quick Start

### Train all 5 classifiers

```bash
bash model/training/train_all_classifiers.sh [DATA_DIR] [EPOCHS] [BATCH_SIZE]
```

Defaults: `data/`, 20 epochs, batch size 8.

### Train one classifier

```bash
python -m model.training.train_classifier \
    --class_name prolongation \
    --data_dir data \
    --epochs 20
```

### Train the CNN localizer

```bash
python -m model.training.train_localizer \
    --data_dir data \
    --epochs 30
```

### Train the Wav2Vec2 localizer

```bash
python -m model.training.train_wav2vec2_localizer \
    --data_dir data \
    --epochs 20 \
    --batch_size 4
```

---

## Classification Pipeline

Trains five independent binary classifiers (one per dysfluency type) on top of `facebook/wav2vec2-base`.

**What it does:** Takes raw audio, passes it through Wav2Vec 2.0, and classifies whether a specific dysfluency is present.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--class_name` | (required) | `prolongation`, `block`, `soundrep`, `wordrep`, or `interjection` |
| `--data_dir` | `data` | Root directory with `audio/` and `labels/` |
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 3e-5 | Learning rate |
| `--model_name` | `facebook/wav2vec2-base` | HuggingFace model |
| `--patience` | 5 | Early stopping patience |
| `--output_dir` | `model/weights` | Where to save weights |

### Training details

- **Split:** Stratified 80/20 train/val split
- **Loss:** BCEWithLogitsLoss with automatic class-weight balancing
- **Optimizer:** AdamW with linear warmup (500 steps) + linear decay
- **Mixed precision:** Automatic on CUDA
- **Checkpointing:** Saves best model (by val F1) and final model
- **Outputs:** `{class}_best.pt`, `{class}_final.pt`, `{class}_training_log.csv`, training curves PNG

---

## CNN Localization Pipeline

Trains a CNN that predicts per-frame dysfluency probabilities from mel-spectrograms.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data` | Root directory with `audio/` and `labels/` |
| `--epochs` | 30 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--cnn_type` | `wrapper` | `wrapper` (CNNSpectrogramLocalizer) or `module` (SpectrogramCNN) |
| `--n_mels` | 128 | Mel frequency bins |
| `--patience` | 7 | Early stopping patience |

### Training details

- **Split:** Random 80/20 train/val
- **Loss:** BCEWithLogitsLoss with `pos_weight=5.0` (dysfluent frames are rare)
- **Optimizer:** AdamW with cosine annealing
- **Metrics:** Frame-level F1, event-level mean IoU

---

## Wav2Vec2 Localization Pipeline

Trains Wav2Vec 2.0 backbone + temporal attention head for per-frame dysfluency prediction from raw audio.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data` | Root directory with `audio/` and `labels/` |
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 4 | Batch size (smaller due to W2V2 memory) |
| `--lr` | 3e-5 | Learning rate |
| `--freeze_backbone_epochs` | 5 | Freeze W2V2 backbone for first N epochs |
| `--patience` | 5 | Early stopping patience |

### Training details

- **Backbone freezing:** Freezes Wav2Vec2 for first N epochs, then unfreezes with 10x lower LR
- **Split:** Random 80/20 train/val
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** AdamW with linear warmup + cosine decay
- **Metrics:** Frame-level F1

---

## Output Structure

```
model/weights/
├── prolongation_best.pt
├── prolongation_final.pt
├── prolongation_training_log.csv
├── block_best.pt
├── block_final.pt
├── block_training_log.csv
├── soundrep_best.pt
├── soundrep_final.pt
├── soundrep_training_log.csv
├── wordrep_best.pt
├── wordrep_final.pt
├── wordrep_training_log.csv
├── interjection_best.pt
├── interjection_final.pt
├── interjection_training_log.csv
├── localizer_best.pt
├── localizer_final.pt
├── localization_training_log.csv
├── w2v2_localizer_best.pt
├── w2v2_localizer_final.pt
├── w2v2_localization_training_log.csv
└── training_curves/
    ├── prolongation_curves.png
    ├── block_curves.png
    ├── soundrep_curves.png
    ├── wordrep_curves.png
    ├── interjection_curves.png
    ├── localization_curves.png
    └── w2v2_localization_curves.png
```

---

## Default Hyperparameters

Defined in `model/config/defaults.py`:

```python
SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 10
MAX_AUDIO_LENGTH = 160000        # samples
WAV2VEC2_BASE = "facebook/wav2vec2-base"
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
NUM_EPOCHS = 20
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_MAX_NORM = 1.0
```

---

## Shared Utilities

`utils.py` provides:

| Utility | Description |
|---------|-------------|
| `save_checkpoint()` / `load_checkpoint()` | Save/load model, optimizer, scheduler state |
| `CSVLogger` | Append-only CSV logging for training metrics |
| `EarlyStopping` | Patience-based early stopping monitor |
| `get_warmup_linear_schedule()` | Linear warmup + linear decay LR schedule |
| `format_duration()` | HH:MM:SS formatting for training time |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No samples found` | Run `python -m model.data.setup` first |
| `CUDA out of memory` | Reduce `--batch_size` (try 4 or 2) |
| `ModuleNotFoundError: model.xxx` | Run from project root, not from `model/` |
| Training too slow | Use GPU, reduce `--num_workers`, or reduce `--max_length_seconds` |
| val F1 stays at 0.0 | Check labels exist in `data/labels/` — see data README |
