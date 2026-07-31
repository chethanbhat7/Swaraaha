# Debugging Log — Wav2Vec2 Training Pipeline

A chronological record of every issue encountered and fix applied while
bringing the training pipeline from silent-failure / F1=0 to a working state.

---

## 1. Label case-mismatch

**Symptom:** `val_acc=1.000, val_F1=0.000` — labels all zeros despite audio
having dysfluencies.

**Root cause:** SEP-28K / UCLASS label CSVs had capitalized dysfluency types
(`Prolongation`, `Block`, `SoundRep`, …) but `CLASS_TO_IDX` in
`model/data/dataset.py` was keyed with lowercase names. `load_label_csv()`
read the raw type string, looked it up in `CLASS_TO_IDX`, found no match,
and produced a zero vector for every sample.

**Files affected:**
- `model/data/dataset.py` — `load_label_csv()`
- `model/data/merge.py` — label merging code

**Fix:** Added `.strip().lower()` to the dysfluency-type field in
`load_label_csv()` and `merge.py`.

**When:** Before the F1=0 training bug investigation.

---

## 2. README Overhauls

**Symptom:** Outdated documentation referencing deleted modules (`Dataset/`)
and missing documentation for new pipelines (orchestrator, augmentation,
preprocessing).

**Files affected:**
- `model/data/README.md`
- `model/training/README.md`

**Changes:**
- Dataset README: replaced stale `Dataset/` references, documented
  `status.py`, `augmentation.py`, `preprocessing.py`, added current
  `data/` directory layout, documented empty-WAV filtering.
- Training README: documented the orchestrator (auto-detection, batch-size
  table, `--dry_run`), `--` forwarding, augmentation mention,
  troubleshooting section for all-zero labels.

---

## 3. F1=0 During Training

This was the main investigation. The model would converge to predicting the
same logit for every sample, yielding F1=0 (all predictions negative).

### 3a. AMP NaN Gradients

**Symptom:** Training ran but classifier never updated (`val_loss` flat,
parameters unchanged).

**Root cause:** `torch.amp.autocast` + `GradScaler` caused the classifier
head's gradients to overflow to `Inf`/`NaN` in float16 precision.
`scaler.step()` detected the NaN gradients and silently skipped the
optimizer update. The backbone received no effective updates.

**Fix:** Removed AMP entirely. Training uses full float32 throughout.

### 3b. num_labels=2 for Binary Classification

**Symptom:** Only half of the classifier head was trained.

**Root cause:** `Wav2Vec2ForSequenceClassification` with `num_labels=2`
creates a 2-output `nn.Linear(hidden_size, 2)`. The training code used
`logits[:, 1]` (the second output) and never touched the first. Half the
head's parameters were randomly initialized and never updated, and the
backprop signal to the backbone was unnecessarily diluted.

**Fix:** Changed `num_labels=2` → `num_labels=1`. Now the classifier is
`nn.Linear(hidden_size, 1)` — a single logit per sample. Removed all
`[:, 1]` indexing; using `logits.squeeze(-1)` instead. Updated `predict()`
to use `torch.sigmoid` (not `softmax`).

**Files affected:**
- `model/classification/__init__.py`
- `model/training/train_classifier.py`
- `model/evaluation/evaluate.py`
- `model/classification/hybrid.py`

### 3c. Classifier Bias Initialisation

**Symptom:** The classifier bias starts at a random value (PyTorch default),
which for imbalanced data can be far from the optimal base-rate logit. The
early gradient signal is weak because the model has to first shift the bias
to the correct range.

**Fix:** Initialise `classifier.bias` to `log(pos/neg)` — the logit
corresponding to the empirical positive rate. For `prolongation`
(~25.5% positive), this is `log(0.255/0.745) ≈ -1.07`.

**Later reverted in favour of freezing bias at 0** (see 3e).

### 3d. Balanced Sampling (WeightedRandomSampler)

**Symptom:** With imbalanced minibatches (~25% positive, ~75% negative), the
model can achieve 74.5% validation accuracy by simply predicting all
negatives. The average gradient at this equilibrium is zero, so the
classifier weight W never learns to use backbone features.

**Root cause:** BCEWithLogitsLoss with imbalanced batches has a stationary
point where the optimal constant prediction (σ = positive rate) gives zero
average gradient. The backbone gradients through W are negligible, so no
feature learning occurs.

**Fix:** Replace uniform batching with a `WeightedRandomSampler` that
oversamples the minority class so each batch has ~50% positive, ~50%
negative samples. This shifts the optimal constant to σ = 0.5 (logit = 0),
right at the decision threshold.

**Effect:** F1 improved from 0.000 to 0.406 — the model now predicts every
sample at the threshold (some above, some below due to floating-point noise)
instead of collapsing to all-negatives. But it still doesn't *learn
features* — it just saturates at the balanced-batch equilibrium.

### 3e. Freezing Classifier Bias at 0

**Symptom:** Even with balanced sampling, the model settles at σ = 0.5 for
every sample. The classifier weight W ≈ 0, bias b ≈ 0, and the backbone
receives zero average gradient. The model is in a different equilibrium but
still not learning features.

**Root cause:** With a trainable bias, the optimal solution is always "set
bias to the optimal constant, set W ≈ 0." The loss landscape has a
degenerate minimum where features contribute nothing. The gradient for W
through the backbone is zero on average at this point, so no escape.

**Fix:** Pin `classifier.bias = 0` and set `requires_grad_(False)`. With the
bias frozen, the model *cannot* predict a constant output. The logit is
purely `W · features`, and the only way to reduce loss is to learn a
non-zero W. This creates a non-zero gradient path from the backbone through
W, enabling feature learning.

### 3f. AMP re-enabled without GradScaler

**Symptom:** Epoch time ~1220s with batch_size=16 on RTX 4070 (20 epochs ≈
7 hours).

**Root cause:** Removing AMP entirely was overly conservative. The NaN
gradient issue was caused by `GradScaler` — it multiplies the loss by 2¹⁶
before backward, pushing classifier gradients past float16's max
representable value (65k). Without the scaler, autocast alone keeps
gradients in float16 range (typical magnitude 10⁻³–10¹).

**Fix:** Re-added `torch.amp.autocast("cuda")` around the forward pass but
removed `GradScaler` entirely. The RTX 4070's Tensor Cores run float16
matmuls at ~2× the throughput of float32. The classifier head stays in
float32 automatically (autocast policy keeps output layers in fp32).

---

## 4. Training Speed Optimisations

### 4a. TF32 Matmul Precision

**Symptom:** Even without AMP, float32 matmuls on RTX 4070 can use TF32
tensor cores (compute capability ≥ 8.0), but PyTorch defaults to "highest"
precision mode which disables this.

**Fix:** Added `torch.set_float32_matmul_precision("high")` at the start of
`train()`. This enables TF32 tensor cores for float32 matrix multiplies,
giving ~8× raw TOPS vs FP32 with no measurable accuracy loss for training.

**File:** `model/training/train_classifier.py`

### 4b. Increased DataLoader Workers

**Symptom:** Audio loading was sequential (`num_workers=0`) despite 24 CPU
cores being available.

**Fix:** Changed default `--num_workers` from 0 to 4 (capped in the
orchestrator at `min(cpu_count, 4)`). Four workers parallelise the
`load_audio()` → `clean_audio()` → `pad_to_length()` pipeline.

**Files:** `model/training/train_classifier.py`, `model/training/train.py`

**Combined effect:** Epoch time reduced from ~1220s to an estimated
700–800s (AMP + TF32 + 4 workers).

---

### 3g. Bias Freeze + Balanced Sampling — Partial Collapse

**Symptom:** With bias frozen at 0 and balanced (50/50) batches, the model
learns for 1-2 epochs (val_F1=0.55) then collapses back (val_F1≈0.34).
Training loss goes UP after epoch 2, indicating unlearning.

**Root cause:** With 50/50 balanced batches and no pos_weight, the optimal
constant prediction is σ=0.5 (logit=0). With bias frozen at 0, the logit is
purely W·features. The model finds W=0 gives σ=0.5 for all samples, which
is a stable equilibrium — the gradient is zero on average. Random init
provides a brief escape (epoch 1-2), but the pull toward W=0 dominates.

**Resolution:** Reverted to standard recipe: imbalanced batches + pos_weight
in BCEWithLogitsLoss, no bias freeze, no balanced sampler.

---

### 3h. BCEWithLogitsLoss — Degenerate Equilibrium (KEY FINDING)

**Symptom:** Every attempt with BCEWithLogitsLoss (sigmoid-based binary
classification) eventually collapses to F1=0. Whether using balanced
sampling, pos_weight, bias freeze, or combinations, the model always finds
a constant-prediction equilibrium and stops learning features.

**Root cause:** BCEWithLogitsLoss with a single logit has a fundamental
degeneracy: the optimal constant prediction (σ = p·w / (1-p + p·w)) gives
**zero average gradient**. With pos_weight = n_neg/n_pos, this optimal
constant is exactly σ = 0.5. At this point, individual sample gradients
average to zero across the batch, so the backbone never receives a useful
learning signal. The model can (and does) achieve its minimum loss without
using any audio features.

This is not a coding bug — it is a property of the loss function. The
single-logit BCE formulation has a flat direction in parameter space that
absorbs all the gradient signal.

**Fix:** Switch to CrossEntropyLoss with num_labels=2 (two logits: one for
"not present", one for "present"). With two logits competing via softmax,
the gradient at the equilibrium (l₀ = l₁, softmax = [0.5, 0.5]) is
**non-zero on average**:

    dL/dl₁ = softmax₁ - y → avg = 0.245 (pushes logit₁ UP, toward minority)
    dL/dl₀ = softmax₀ - (1-y) → avg = -0.245 (pushes logit₀ DOWN)

This asymmetric gradient breaks the deadlock. The model is mathematically
forced to discriminate between classes. This is the standard HuggingFace
recipe used by Wav2Vec2 papers achieving F1=0.75+ on SEP-28K.

**Files affected:**
- `model/classification/__init__.py` — `num_labels=2`, `softmax` in predict
- `model/training/train_classifier.py` — CrossEntropyLoss, `.long()` labels
- `model/evaluation/evaluate.py` — `softmax`, `probs[:, 1]` for scores
- `model/classification/hybrid.py` — `logits[:, 1]` (present logit)

---

### 3i. Gradient Clipping Destroying Learning

**Symptom:** With CrossEntropyLoss + class weights, first epoch loss stays
at 0.693 (random baseline). Model never learns. With or without weights,
results are the same — constant prediction, F1=0.

**Root cause:** `clip_grad_norm_(parameters, max_norm=1.0)` was applied to
ALL model parameters (94.6M total). The randomly initialized classifier
head (1538 params) has large gradients (~100 norm) at the start of training.
The backbone (94.6M params) has tiny gradients (~0.1 norm). After clipping
the total gradient norm to 1.0, the classifier head's gradients are reduced
100×, and the backbone gradients become vanishingly small. Neither learns.

**Fix:** Removed `clip_grad_norm_` entirely. AdamW's built-in regularization
(weight_decay) is sufficient for stability. The classifier head can now
update at full magnitude, carrying a usable gradient signal to the backbone.

---

### 3j. Class Weights Create Zero-Gradient Equilibrium (CrossEntropy version)

**Symptom:** With CrossEntropyLoss + class weights [1.0, 2.92] (inverse frequency
weighting), loss stays at exactly 0.693 = -ln(0.5) for 4+ epochs. F1=0. Model
never moves from uniform prediction.

**Root cause:** The weight `w₁ = n_neg/n_pos = (1-r)/r` creates an exact
zero-gradient equilibrium at the uniform output point (logits=[0,0],
softmax=[0.5, 0.5]):

```
dL/d(logit₀) = (r·w₁ − (1−r)·w₀) · 0.5
dL/d(logit₁) = ((1−r)·w₀ − r·w₁) · 0.5

With r=0.255, w₀=1.0, w₁=2.92:  dL = [0, 0] exactly
```

This is the same degeneracy as BCE + pos_weight (3h), just in the 2-logit
CrossEntropy formulation. The weights perfectly balance the class gradient
contributions, so at 50/50 output every sample's gradient cancels its
counterpart. The model is **not stuck due to insufficient gradient magnitude**
— it is stuck because the expected gradient is mathematically zero.

Additionally, CrossEntropyLoss with `reduction='mean'` and `weight` normalizes
by `sum(weight[target])`, not by batch size. This means the loss value at
uniform output is always 0.693 regardless of class distribution, masking the
problem.

**Fix:** Two changes:
1. **FocalLoss** (γ=2.0) replaces CrossEntropyLoss. Focal loss downweights
   easy examples via `(1-p_t)^γ`. At uniform output (p_t=0.5), the gradient
   for positive and negative samples is NOT equal in magnitude:
   negative gradient = -0.075, positive gradient = +0.075 (vs CrossEntropy's
   ±0.5). While the direction still favours the majority class, Focal loss
   naturally focuses training on minority (positive) samples as the model
   becomes confident about negatives.
2. **Freeze backbone for 2 epochs** — Train only the randomly-initialized
   classifier head (1538 params) on the fixed Wav2Vec2 features. This lets
   the head learn the correct logit mapping before the 94.6M backbone params
   are exposed to gradients. After unfreezing, the backbone fine-tunes with
   a head that already produces useful logits.

**Files affected:**
- `model/training/utils.py` — Added `FocalLoss` class
- `model/training/train_classifier.py` — Added `--freeze_backbone_epochs`,
  `--loss_type`, `--focal_gamma` args; freeze/unfreeze logic; removed class
  weights

---

## Current State

The training pipeline on `feat/training-bugfix` includes all fixes above.
The current recipe uses FocalLoss + backbone freezing:

```bash
python -m model.training.train_classifier \
    --class_name prolongation \
    --data_dir data/train \
    --epochs 20 \
    --batch_size 16 \
    --lr 3e-5 \
    --freeze_backbone_epochs 2 \
    --loss_type focal \
    --focal_gamma 2.0 \
    --output_dir model/weights
```

Key design decisions:
- **FocalLoss** (γ=2.0) — breaks zero-gradient equilibrium without perfect
  class weighting
- **Freeze backbone for 2 epochs** — head-only training lets the classifier
  stabilize before backbone fine-tuning
- **No class weights** — FocalLoss handles imbalance implicitly
- **No gradient clipping** — was silently killing learning (3i)
- **AMP with autocast (no GradScaler)** — safe float16 for speed
- **Imbalanced batching** — natural distribution, no balanced sampler

### Summary of Changes

| # | Fix | File(s) | Why |
|---|-----|---------|-----|
| 1 | `.strip().lower()` in label parsing | `dataset.py`, `merge.py` | Capitalisation mismatch |
| 2 | README updates | `model/data/README.md`, `model/training/README.md` | Stale docs |
| 3a | Remove AMP | `train_classifier.py` | NaN grads from float16 |
| 3b | `num_labels=1` → `num_labels=2` | `__init__.py`, `train_classifier.py`, `evaluate.py`, `hybrid.py` | Single logit has zero-gradient equilibrium |
| 3c | Bias init → `log(pos/neg)` | `train_classifier.py` | Faster convergence |
| 3d | `WeightedRandomSampler` | `train_classifier.py` | Balanced batches prevent all-negative cheat |
| 3e | Freeze bias at 0 | `train_classifier.py` | Forces feature learning through W |
| 3f | Re-add AMP (no GradScaler) | `train_classifier.py` | NaN was from scaler, not autocast |
| 3g | Revert to imbalanced + pos_weight | `train_classifier.py` | Bias freeze + balanced sampling caused W=0 collapse |
| 3h | BCE → CrossEntropyLoss + num_labels=2 | `__init__.py`, `train_classifier.py`, `evaluate.py`, `hybrid.py` | BCE has degenerate zero-gradient equilibrium |
| 3i | Remove gradient clipping | `train_classifier.py` | `max_norm=1.0` killed learning by scaling head grads 100× |
| 3j | FocalLoss + backbone freezing | `utils.py`, `train_classifier.py` | Class weights also create zero-gradient equilibrium |
| 4a | TF32 matmul precision | `train_classifier.py` | ~8× TOPS on matmuls, free speed |
| 4b | DataLoader `num_workers=4` | `train_classifier.py`, `train.py` | Parallel audio loading |
| 4c | Audio preprocessing cache | `dataset.py` | Pickle cache for load_audio → clean_audio → pad_to_length pipeline |
| 4d | Auto `num_workers` from CPU count | `train_classifier.py` | Removed cap at 4, now uses `os.cpu_count()` |
| 4e | Gradient accumulation | `train_classifier.py` | `--gradient_accumulation_steps` for effective batch size scaling |
| 4f | `torch.compile` | `train_classifier.py` | JIT-compile model on CUDA via `torch.compile` |

### 4g. `torch.compile` Property Assignment

**Symptom:** `AttributeError` — `model.model = torch.compile(model.model)` failed because `model.model` is a read-only `@property` that returns `self._model`.

**Fix:** Assign to `model._model` directly: `model._model = torch.compile(model._model)`.

**File:** `train_classifier.py`

---

## 5. Unfreeze — Differential LR & Optimizer State

**Symptom:** When backbone unfreezes after epoch N, the old code re-created the optimizer from scratch. This (a) lost AdamW momentum from head-only training, and (b) applied the same LR to all params.

**Fix:** Instead of recreating the optimizer, add a second param group with `lr * 0.1` for backbone params:

```python
optimizer.add_param_group({"params": backbone_params, "lr": args.lr * 0.1})
scheduler.base_lrs.append(args.lr * 0.1)
scheduler.lr_lambdas.append(scheduler.lr_lambdas[0])
```

This preserves the head's optimizer state and applies a 10× smaller LR to backbone params.

**File:** `train_classifier.py`

---

## 6. Training Resume

**Symptom:** An interrupted training run (Ctrl+C, crash) lost all progress. No way to resume from the last completed epoch.

**Fix:** Added checkpoint-based resume. After each epoch, a `{class}_checkpoint.pt` file is saved containing: model, optimizer, scheduler, epoch, history, best_F1, backbone_frozen flag, and CLI args. On startup, `_try_load_resume()` checks:
  - If checkpoint exists AND saved args match current args → restore state, continue training
  - If args differ → start fresh (config changed, old checkpoint invalid)
  - If `--clean` flag → ignore checkpoint, force fresh start

**Files:** `train_classifier.py`, `utils.py`
**Functions:** `save_resume_checkpoint()`, `load_resume_checkpoint()`, `args_match()`

---

## Current State

The training pipeline on `feat/training-resume` includes all fixes above.

```bash
python -m model.training.train_classifier \
    --class_name prolongation \
    --data_dir data/train \
    --epochs 20 \
    --batch_size 16 \
    --lr 3e-5 \
    --freeze_backbone_epochs 2 \
    --loss_type focal \
    --focal_gamma 2.0 \
    --gradient_accumulation_steps 1 \
    --num_workers auto \
    --output_dir model/weights
```

Key design decisions:
- **Audio cache** — preprocessed audio cached to `data/cache/{split}` pickle files; subsequent runs load in ~1s
- **Auto `num_workers`** — uses all CPU cores for data loading
- **`torch.compile`** — JIT-compiles model on CUDA for ~2× training speed
- **Gradient accumulation** — scales effective batch size without memory increase
- **Stable unfreeze** — preserves optimizer momentum, applies 10× smaller LR to backbone
- **Training resume** — checkpoint-based interruption recovery with args validation

### Summary of Changes

| # | Fix | File(s) | Why |
|---|-----|---------|-----|
| 1 | `.strip().lower()` in label parsing | `dataset.py`, `merge.py` | Capitalisation mismatch |
| 2 | README updates | `model/data/README.md`, `model/training/README.md` | Stale docs |
| 3a | Remove AMP | `train_classifier.py` | NaN grads from float16 |
| 3b | `num_labels=1` → `num_labels=2` | `__init__.py`, `train_classifier.py`, `evaluate.py`, `hybrid.py` | Single logit has zero-gradient equilibrium |
| 3c | Bias init → `log(pos/neg)` | `train_classifier.py` | Faster convergence |
| 3d | `WeightedRandomSampler` | `train_classifier.py` | Balanced batches prevent all-negative cheat |
| 3e | Freeze bias at 0 | `train_classifier.py` | Forces feature learning through W |
| 3f | Re-add AMP (no GradScaler) | `train_classifier.py` | NaN was from scaler, not autocast |
| 3g | Revert to imbalanced + pos_weight | `train_classifier.py` | Bias freeze + balanced sampling caused W=0 collapse |
| 3h | BCE → CrossEntropyLoss + num_labels=2 | `__init__.py`, `train_classifier.py`, `evaluate.py`, `hybrid.py` | BCE has degenerate zero-gradient equilibrium |
| 3i | Remove gradient clipping | `train_classifier.py` | `max_norm=1.0` killed learning by scaling head grads 100× |
| 3j | FocalLoss + backbone freezing | `utils.py`, `train_classifier.py` | Class weights also create zero-gradient equilibrium |
| 4a | TF32 matmul precision | `train_classifier.py` | ~8× TOPS on matmuls, free speed |
| 4b | DataLoader `num_workers=4` | `train_classifier.py`, `train.py` | Parallel audio loading |
| 4c | Audio preprocessing cache | `dataset.py` | Avoid re-processing audio every run |
| 4d | Auto `num_workers` from CPU count | `train_classifier.py` | No hard cap, uses `os.cpu_count()` |
| 4e | Gradient accumulation | `train_classifier.py` | Effective batch size scaling |
| 4f | `torch.compile` | `train_classifier.py` | JIT compilation on CUDA |
| 4g | `torch.compile` property fix | `train_classifier.py` | `model.model` is read-only property |
| 5 | Stable unfreeze (preserve optimizer, 10× LR) | `train_classifier.py` | Loses momentum when recreating optimizer |
| 6 | Training resume checkpoint | `train_classifier.py`, `utils.py` | Interruption recovery |
