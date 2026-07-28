# Training Pipelines

---

## Prerequisites

```bash
pip install -r model/requirements.txt
```

Prepare the dataset first (see [`model/data/README.md`](../data/README.md)):

```bash
python -m model.data.setup
```

---

## Quick Start

Train everything with auto-detected system resources:

```bash
python -m model.training.train
```

This detects your GPU (RTX 4070 → batch size 16, etc.) and trains all five classifiers plus both localizers.

Select specific pipelines:

```bash
python -m model.training.train --pipelines cls          # classifiers only
python -m model.training.train --pipelines loc          # CNN localizer only
python -m model.training.train --pipelines wav2vec      # Wav2Vec2 localizer only
python -m model.training.train --pipelines cls loc      # multiple
```

Dry run to see what would be run:

```bash
python -m model.training.train --dry_run
```

### Train one classifier

```bash
python -m model.training.train_classifier \
    --class_name prolongation \
    --data_dir data/train \
    --epochs 20
```

### Train all 5 classifiers via shell script

```bash
bash model/training/train_all_classifiers.sh
```

---

## Training Orchestrator

`model/training/train.py` is the master orchestrator that:

1. **Auto-detects system resources** — CPU cores, GPU model, GPU memory
2. **Selects optimal batch sizes** — based on GPU memory:
   | Pipeline | 8+ GB GPU | <8 GB GPU | CPU |
   |----------|-----------|-----------|-----|
   | Classifier | 16 | 8 | 4 |
   | CNN Localizer | 8 | 4 | 2 |
   | Wav2Vec2 Localizer | 4 | 2 | 1 |
3. **Sets DataLoader workers** to CPU core count
4. **Runs each pipeline** sequentially with proper CLI args

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--pipelines` | `cls loc wav2vec` | Which pipelines to run |
| `--data_dir` | `data/train` | Training data directory |
| `--output_dir` | `model/weights` | Where to save weights |
| `--dry_run` | false | Show what would run without executing |

Any arguments after `--` are forwarded to each sub-script (train_classifier, train_localizer, etc.):

```bash
python -m model.training.train -- --epochs 30 --lr 2e-5
python -m model.training.train --pipelines cls -- --batch_size 8
```

---

## Classification Pipeline

Trains five independent binary classifiers (one per dysfluency type) on top of `facebook/wav2vec2-base`.

**What it does:** Takes raw audio, passes it through Wav2Vec 2.0, and classifies whether a specific dysfluency is present.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--class_name` | (required) | `prolongation`, `block`, `soundrep`, `wordrep`, or `interjection` |
| `--data_dir` | `data/train` | Root directory with `audio/` and `labels/` |
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 3e-5 | Learning rate |
| `--model_name` | `facebook/wav2vec2-base` | HuggingFace model |
| `--patience` | 5 | Early stopping patience |
| `--output_dir` | `model/weights` | Where to save weights |
| `--num_workers` | 0 | DataLoader workers |

### Training details

- **Split:** Stratified 80/20 train/val split per class
- **Loss:** BCEWithLogitsLoss with automatic class-weight balancing
- **Optimizer:** AdamW with linear warmup (500 steps) + linear decay
- **Mixed precision:** Automatic on CUDA via `torch.amp.GradScaler`
- **Augmentation:** On-the-fly `AudioAugmentor` (noise, pitch shift, time stretch, roll, scale)
- **Checkpointing:** Saves best model (by val F1) and final model
- **Outputs:** `{class}_best.pt`, `{class}_final.pt`, `{class}_training_log.csv`, training curves PNG

---

## CNN Localization Pipeline

Trains a CNN that predicts per-frame dysfluency probabilities from mel-spectrograms.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data/train` | Root directory with `audio/` and `labels/` |
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
| `--data_dir` | `data/train` | Root directory with `audio/` and `labels/` |
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
MAX_AUDIO_LENGTH = 160000
WAV2VEC2_BASE = "facebook/wav2vec2-base"
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
NUM_EPOCHS = 20
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
AUGMENTATION_ENABLED = True
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
| `get_warmup_linear_schedule()` | Linear warmup + linear decay LR schedule using `transformers` scheduler |
| `format_duration()` | HH:MM:SS formatting for training time |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No samples found` | Run `python -m model.data.setup` first |
| `CUDA out of memory` | Reduce `--batch_size` (try 4 or 2), or use the orchestrator which auto-tunes |
| `ModuleNotFoundError: model.xxx` | Run from project root, not from `model/` |
| Training too slow | Use GPU, reduce `--num_workers`, or reduce `--max_length_seconds` |
| `val F1 stays at 0.0` | Check that labels use lowercase types (`prolongation` not `Prolongation`) in `data/train/labels/*.csv` |
| Train loss drops to ~0 instantly, val_acc=1.0 | Labels likely all zeros — check the interval CSV format |
