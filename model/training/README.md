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

This detects your GPU and trains all five classifiers plus both localizers.

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
| `--num_workers` | `auto` | DataLoader workers (`auto` = CPU core count) |
| `--freeze_backbone_epochs` | 3 | Freeze W2V2 backbone for first N epochs (train head only) |
| `--gradient_accumulation_steps` | 1 | Accumulate gradients over N batches (effective BS = BS × steps) |
| `--loss_type` | `focal` | Loss function: `focal` or `cross_entropy` |
| `--focal_gamma` | 2.0 | Focal loss focusing parameter (only if `--loss_type=focal`) |
| `--warmup_steps` | 500 | Linear LR warmup steps |
| `--clean` | false | Ignore resume checkpoint and start training from scratch |

### Training details

- **Split:** Stratified 80/20 train/val split per class
- **Loss:** Focal loss (γ=2.0) for better handling of class imbalance
- **Optimizer:** AdamW with linear warmup (500 steps) + linear decay
- **Mixed precision:** Automatic on CUDA via `torch.amp.GradScaler`
- **Augmentation:** On-the-fly `AudioAugmentor` (noise, pitch shift, time stretch, roll, scale)
- **Audio cache:** Preprocessed audio is cached to `data/cache/{split}` (pickle) — subsequent runs load instantly
- **`torch.compile`:** Model is JIT-compiled via `torch.compile` when on CUDA for faster training
- **Checkpointing:** Saves best model (by val F1), final model, and resume checkpoint (model + optimizer + scheduler + args)

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

All outputs use **parameter-specific fingerprint naming**. The filename encodes every hyperparameter:

```
{class}_e{epochs}_b{batch_size}_lr{lr}_frz{freeze_backbone_epochs}_{loss_type}_g{focal_gamma}_ga{gradient_accumulation_steps}_wu{warmup_steps}_wd{weight_decay}_ml{max_length_seconds}_s{seed}_{data_short}_{model_short}_{suffix}.pt
```

Example:
```
prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base_best.pt
```

Suffixes: `_best.pt` (best val F1), `_final.pt` (last epoch), `_checkpoint.pt` (resume state), `_log.csv` (training metrics), `_curves.png` (training curves), `_training.log` (terminal output).

```
model/weights/
├── prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base_best.pt
├── prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base_final.pt
├── prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base_checkpoint.pt
├── prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base_log.csv
├── prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base_training.log
├── ... (same pattern for block, soundrep, wordrep, interjection)
└── training_curves/
    ├── prolongation_e20_b8_..._curves.png
    └── ...
```

To use trained models, register them in `model/registry.json`:

```json
{
  "classification": {
    "prolongation": "model/weights/prolongation_e20_b8_lr3e-5_..._best.pt",
    "block": "model/weights/block_e20_b8_lr3e-5_..._best.pt",
    ...
  },
  "thresholds": {
    "prolongation": 0.5,
    "block": 0.5,
    ...
  }
}
```

Then access via the registry API — audio (path/bytes/numpy) is preprocessed automatically:

```python
from model import Classifier
result = Classifier().analyze("recording.wav")      # all 5 + summary
result = Classifier().analyze_raw("recording.wav")  # + logits
result = Classifier("prolongation").analyze(audio)  # single class
```

---

## Training Resume

Training auto-saves a resume checkpoint after every epoch. If interrupted (Ctrl+C, crash), re-running the **same command** picks up from the last completed epoch:

```bash
python -m model.training.train_classifier --class_name prolongation
# ^C after epoch 3
python -m model.training.train_classifier --class_name prolongation  # resumes at epoch 4
```

If you change any flag (e.g., `--lr`), the checkpoint is invalidated — training starts fresh. Use `--clean` to force a fresh start:

```bash
python -m model.training.train_classifier --class_name prolongation --clean
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
| `save_checkpoint()` / `load_checkpoint()` | Save/load model, optimizer, scheduler, history, and CLI args |
| `CSVLogger` | Append-only CSV logging for training metrics |
| `save_resume_checkpoint()` / `load_resume_checkpoint()` | Full training state for interruption-safe resume |
| `args_match()` | Compares saved vs current CLI args to detect config changes |
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
