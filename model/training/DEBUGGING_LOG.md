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

The training pipeline on `feat/model-registry` includes all fixes above.
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
- **Audio cache** — preprocessed audio cached to `data/cache/{split}` pickle
- **`torch.compile`** — JIT-compiles model on CUDA for ~2× speed
- **Gradient accumulation** — effective batch size scaling
- **Stable unfreeze** — preserves optimizer momentum, 10× smaller LR to backbone
- **Training resume** — checkpoint-based interruption recovery
- **Model registry** — `Classifier()`, `Localizer()`, `ModelRegistry()` API for loading

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

The training pipeline on `feat/model-registry` includes all fixes above.

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

All 5 classifiers trained. Model registry (`model/registry.py`) provides the API
for loading trained models. Use `Classifier()`, `Localizer()`, or `ModelRegistry()`
— do not import model classes directly.

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

---

## 7. First Full Run — Results (freeze=3, 20 epochs)

**Best val F1: 0.5239** (epoch 19) | Final F1: 0.522 | Best val acc: 0.773

```
Epoch  1-3  | head-only      | F1 0.373 → 0.390          (head converges on frozen features)
Epoch  4-5  | UNFROZEN       | F1 0.000 → 0.003          (collapse, ~2 wasted epochs)
Epoch  6-16 | recovery+climb | F1 0.399 → 0.520          (breaks out, learns features)
Epoch 17-20 | plateau        | F1 0.512 → 0.522 → 0.524   (LR decays to 0)
```

**Interpretation:**
- The unfreeze collapse is now a **temporary transition**, not a dead end — the model
  reliably recovers within 1-2 epochs and surpasses its pre-collapse peak.
- `val_acc` plateaus at ~0.75, which is close to the negative base rate (74.5%).
  The model is still conservative (predicts mostly negative) but achieves F1≈0.52 via
  precise positive hits. Typical SEP-28K wav2vec2 results are ~0.75 F1, but this
  val set mixes three datasets (Boli + SEP-28K + UCLASS), so 0.52 is a reasonable
  first-pass number, not a bug.
- At epoch 20 the LR hits exactly `0.00e+00` — the linear decay schedule is exhausted.
  **Running more epochs with the same schedule does nothing** (zero update signal).

---

## 8. Future Improvements (not yet implemented)

### 8a. Warm restart after LR exhaustion

Since LR hits 0 at the end of the schedule, the model is mathematically frozen.
A warm restart re-inflates LR (e.g. back to `3e-5`) and runs a second decay cycle,
letting the optimizer escape the plateau. Cosine annealing with restarts (SGDR)
or simply re-running with `--clean` at a higher `--epochs` are both easy paths.

### 8b. Reduce the unfreeze collapse (epochs 4-5)

Two epochs (~1500s each) are wasted on the post-unfreeze collapse. Candidate fixes:
- **Lower backbone LR** — try `lr * 0.01` instead of `0.1` so the backbone barely
  moves at first, giving the head time to adapt to changing features.
- **Gradual / layer-wise unfreeze** — unfreeze the last transformer layer first,
  then progressively earlier layers (standard for fine-tuning LLMs/ASR).
- **Backbone-specific warmup** — add a separate warmup for the new backbone param
  group so its effective LR ramps from ~0 instead of jumping in at 3e-6.

Eliminating these 2 dead epochs would push the effective ceiling toward ~0.55+.

### 8c. ~~Train the other four classifiers~~ ✅ Done

All 5 classifiers trained. Results (val F1):
- prolongation: 0.5239, block: 0.5288, soundrep: 0.0019, wordrep: 0.0135, interjection: 0.6830

### 8d. ~~Integrate trained weights into the app~~ ✅ Done

Solved via the **model registry** (`model/registry.py` + `model/registry.json`).
The backend service (`backend/services/classifier.py`) now uses `Classifier()` from the registry.
`Classifier.analyze()` accepts raw audio (path/bytes/numpy), applies preprocessing automatically,
and returns label/confidence/probabilities (plus logits via `analyze_raw`). The hybrid combiner
is not used — all-5 output is raw per-classifier results plus a detected-classes summary.

### 8e. Bigger model or pretrained ASR features

`facebook/wav2vec2-base` (94M params) is the small variant. `wav2vec2-large`
(315M) or a whisper-encoder baseline typically adds a few points of F1 at
~3× training time.

### 8f. Threshold tuning

F1 is optimized at the default 0.5 decision threshold. Sweeping the threshold on
the val set (precision/recall curve) can recover several points of F1 with zero
retraining — worth doing before any architecture change.

---

## 9. Localizer: waveform augmentation applied to spectrograms

**Symptom:** CNN spectrogram localizer frame `F1` stuck at `0.000`; validation
loss dropped but no frame ever crossed the `0.5` threshold.

**Root cause:** `AugmentedDataset` ran the *waveform* `AudioAugmentor` (noise,
time-stretch, pitch-shift, roll) directly on `(1, n_mels, T)` spectrogram
arrays. Stretch/pitch-shift resampled the time axis while frame labels stayed
put — augmented "signal" had near-zero correlation with labels (measured ≈
`−0.11`). Training on misaligned labels ≈ training on noise.

**Files affected:**
- `model/data/augmentation.py`

**Fix (rewrote `model/data/augmentation.py`):**
- `SpectrogramAugmentor` — SpecAugment-style masking on `(C, n_mels, T)`;
  masking never moves energy in time, labels stay valid.
- `AudioAugmentor.apply_with_labels(...)` — samples each transform once and
  applies it to audio AND frame labels in lockstep (stretch/pitch via
  `_resample`, roll shifts labels by `round(shift / frame_hop)` frames).
- `AugmentedDataset` routes by input dimensionality + `label_aligned` /
  `frame_hop_samples` instead of waveform-augmenting everything.

**Verified by:** 12 new tests in `model/data/tests/test_augmentation.py`.

---

## 10. Localizer: SEP-28K weak labels inflate all-source metrics

**Symptom:** All-sources run (sep28k + uclass) hit frame `F1 0.85`, IoU `0.90`
— looked great but was not the trustworthy number.

**Root cause:** SEP-28K intervals span the whole clip (`0.000,3.000`), so
whole-clip predictions trivially match whole-clip ground truth. Only UCLASS has
precise event intervals.

**Fix:** The trustworthy number comes from a UCLASS-only run:
`python -m model.training.train --pipelines loc wav2vec -- --sources uclass --clean`.

---

## 11. Localizer: `pos_weight=5.0` ~8× too weak (current case)

**Symptom:** UCLASS-only CNN run: `AUROC 0.927` but frame `F1@0.5 = 0.011`
(final best `0.014`). Prediction distribution: `max = 0.537`, `p99.9 = 0.488`
— nothing meaningfully exceeds `0.5`. Best-threshold F1 only `0.256`
(at `t = 0.38`); precision at the API default threshold `0.3` is `0.148`
(85% false positives).

**Root cause:** Positive-frame rate on UCLASS is `2.5%`, so the balanced
`pos_weight` should be `n_neg/n_pos ≈ 39`. The hardcoded `5.0` is ~8× too
weak — the loss never pushes outputs into confident territory, and every
prediction is compressed into a low-probability band overlapping the negatives.
High AUROC despite poor F1 = small separable subset drives the ranking.

**Diagnostic evidence** (evaluate on val, `check_cnn_preds.py`):
```
positive ratio = 0.02486
F1@0.5 = 0.0114   AUROC = 0.9273   best-threshold F1 = 0.256 at t=0.38
pos preds: mean 0.384, p90 0.464
neg preds: p99.9 0.484, max 0.537   ← negatives beat most positives
t=0.3 → F1 0.249, precision 0.148, recall 0.793
```

**Fix (implemented, awaiting retrain):** `--pos_weight` default `5.0` → `None`
(auto-computed from training-split frame labels, inverse frequency); best-
threshold F1 reported alongside AUROC. Files:
`model/training/train_localizer.py`, `model/training/train_wav2vec2_localizer.py`,
`model/training/utils.py`.

---

## 12. Localizer: Wav2Vec2 path is the working one

UCLASS-only w2v2 run reached frame `F1@0.5 = 0.322`, `AUROC 0.943` at epoch 6
and still climbing. The pretrained backbone separates events even at threshold
0.5 — no compression problem. Expect CNN to remain the weak link.

---

## 13. Project Boli: all clips silently skipped in merge

**Symptom:** 0 Boli clips ingested; combined labels only had sep28k + uclass.

**Root causes (two, in `model/data/merge.py`):**
1. **Filename mismatch:** transcript stems are `{nEvents}_{speaker}_{task}`
   (`10_727253_EI`) but audio files are `{speaker}_english_{task}_blob.wav`
   (`727253_english_image_blob.wav`). Matcher only tried `{stem}.wav`.
2. **Timestamps are seconds, not ms:** the parser divided by 1000, so every
   Boli interval would be 1000× too small.

**Fix (TDD, 9 tests in `model/data/tests/test_merge.py`):**
`_boli_audio_name()` maps `{nEvents}_{speaker}_{task}` →
`{speaker}_english_{task}_blob.wav` (task codes `E1/E2/E3/EI`); raw seconds
kept as seconds.

**Verified by:** real-data run → 54/55 clips ingested as `source='boli'`
(312 intervals: SR 159, B 74, PR 42, WR 21, IN 16). The 1 skipped clip is a
genuine missing-audio case upstream.

---

## 14. Transcription fidelity: Whisper is not a stutter detector

**Probe:** 100 SEP-28K single-label clips per group (soundrep, wordrep, block,
prolongation, clean) + 50 UCLASS clips with precise events, through the real
`Transcriber.transcribe()` (Whisper-tiny + `_flag_repetitions`).

**Results:**
- Raw fidelity: sound-reps as repeated tokens in **1/100** clips; blocks
  vanish entirely (**0/100**).
- `wordrep` flag fired **0/500** times — even where Whisper DID transcribe the
  repeat.
- `soundrep` flags dominated by ellipsis false positives: regex `(.)\1{2,}`
  matches `...` in `"she..."`, `"just..."`, plus `,000` in numbers.
  False-positive rate on clean clips (4%) > recall on real soundreps (3%).
- UCLASS event-level: recall 6.1%, precision 75% (only 4 flags in 50 clips).

**Root causes:**
1. `_transcribe_with_whisper` de-dups consecutive identical chunks
   (hallucination guard) *before* `_flag_repetitions` runs → genuine repeats
   already collapsed.
2. `_is_repeated_fragment` regex matches 3+ trailing periods.

**Fix:** none applied yet (pending: fix regex + reorder flagging, or document
as display-layer limitation).

**Design implication:** detection never uses the transcript; type-linking stays
audio-only (classifier saliency). Transcription is display-only.

---

## 15. `--sources` not in localizer fingerprint → silent skip

**Symptom:** rerun with different `--sources` would silently skip because
`maybe_skip_completed` saw a completed checkpoint with the same fingerprint.

**Root cause:** `sources` not in `CNN_LOCALIZER_RESUME_KEYS` /
`W2V2_LOCALIZER_RESUME_KEYS`.

**Fix (workflow):** always pass `--clean` when changing sources.

---

## 16. Probe infra bug: `pipe.__call__` monkeypatch ignored

**Symptom:** raw Whisper chunks never captured; raw-fidelity metrics showed 0%.

**Root cause:** Python resolves `__call__` on the *type*, not the instance;
`pipe.__call__ = fn` is silently ignored.

**Fix:** wrap the pipeline in a `_Proxy` instance that stashes raw output on a
side channel.

---

## 17. M13 labels reached `data/labels` but never the training splits

**Symptom:** After re-running `python -m model.data.setup`, retraining produced
the same inflated positive ratios as before the M13 (SEP-28K majority-vote)
fix — training results looked unchanged.

**Root cause:** `setup.py` ran `merge` and `prepare` as subprocesses with no
flags. The re-merge DID write majority-vote labels to `data/labels`, but
`prepare` only copies a label when the destination is missing or `--force` is
given — so existing `data/train/labels` kept the old single-annotator labels.
The new ground truth never reached the training splits.

**Fix (operational):** `python -m model.data.prepare --force`, which
re-copied all 36,674 labels into the splits (verified: block positive ratio
dropped 0.401 → 0.141).

**Fix (code):** `setup.py` now accepts `--force` (forwarded to merge +
prepare) and forwards any `--` extra args to every step, so re-running setup
with `python -m model.data.setup --force` regenerates and propagates labels.

---

## 18. Multitask classifier: `load_multitask` crashed on training-format checkpoints

**Symptom:** `--model_type multitask` eval crashed with
`KeyError: 'model_name'` in `MultiTaskWav2VecClassifier.from_pretrained`
(model/classification/multitask.py:135).

**Root cause:** Format mismatch. Training scripts save via `save_checkpoint`
(model/training/utils.py:17-56) → `{epoch, model_state_dict,
optimizer_state_dict, metrics, scheduler_state_dict}` with **no `model_name`**.
The model class's own `save()` writes `{model_state_dict, model_name,
class_names, hidden_dim}`. `load_multitask` (model/evaluation/loader.py)
unconditionally called `from_pretrained`, which requires `model_name`.
`load_classifier` already handled both formats (checks `"model_name" in
checkpoint`); `load_multitask` did not.

**Fix:** `load_multitask` now checks for `"model_name"` in the checkpoint. If
absent, it infers the HF model name via `model_name_from_path`
(model/fingerprint.py:77, maps `w2v2base` → `facebook/wav2vec2-base`; the
`multi_` fingerprint parses fine), builds `MultiTaskWav2VecClassifier(
model_name=..., hidden_dim=768, class_names=None)`, strips `_orig_mod.`
prefixes, `load_state_dict(strict=True)`.

**Test:** `model/evaluation/tests/test_loader.py:test_load_multitask_training_format_without_model_name`
(monkeypatched model class). Full suite: 296 passed.

---

## 19. First multitask training run — results (freeze=3, focal, 20 epochs)

**Fingerprint:** `multi_e20_b16_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base`
**Artifacts:** `model/weights/{fp}_{best,final,checkpoint}.pt`, `_log.csv`, `_training.log`.

**Result:** 11 epochs (early stop, patience 5, best = epoch 6). val macro-F1
0 → 0.3651 (best), 0.357 at ep11. val_acc 0.87–0.90. Train Focal loss 0.51→0.34.
Backbone frozen 3 epochs, unfroze ep4 (head LR 3e-5, backbone 3e-6). 7395.7s total.

**Decision: `val_loss` (1.7–2.6) is NOT a real problem.** It's a
metric-mismatch artifact: val loss = CrossEntropy summed over 5 heads
(train_multitask_classifier.py:211), train loss = FocalLoss γ=2 summed (:354).
Focal downweights easy examples, so it is systematically ~5× lower.
val_loss was actually decreasing across epochs.

**Decision: MT beats the single-classifier approach on hard classes.** Earlier
b16 single-classifier runs collapsed to all-negative (F1 0.0). MT reaches 0.365
macro. Old single-class block classifier had AUROC 0.499 (≈random,
always-positive); MT block head gets AUROC 0.752 — the shared backbone fixes
the hard classes.

**Eval on best.pt** (data/train, 20% stratified split seed 42, 5857 val
samples, @t=0.5):

| class        | F1    | AUROC | support |
|--------------|-------|-------|---------|
| prolongation | 0.422 | 0.820 | 568     |
| block        | 0.147 | 0.752 | ~832    |
| soundrep     | 0.281 | 0.832 | ~715    |
| wordrep      | 0.249 | 0.756 | ~562    |
| interjection | 0.725 | 0.916 | 1259    |
| **macro avg**| **0.365** |      |         |

Positive ratios (train): prolongation 0.095, block 0.142, soundrep 0.122,
wordrep 0.096, interjection 0.218.

**Observation:** AUROC ≫ F1@0.5 for every class ⇒ threshold mis-calibration is
the bottleneck, not ranking quality. Drives section 20.

---

## 20. Threshold tuning phase (current)

**Decision:** Focus on proving and capturing the per-class threshold gap for the
multitask classifier before any retraining. Rationale: AUROC (0.75–0.92) is far
above what a single fixed 0.5 threshold produces (macro F1 0.365), so the model
ranks well — the operating point is wrong.

**Scope (all in):**
1. `evaluate_multitask` honors `--sweep_thresholds`.
2. Per-class optimal thresholds persisted to `multitask_thresholds.json`.
3. `MultiTaskClassifier.analyze` (registry) applies per-class thresholds.

**Design doc:** `docs/superpowers/specs/2026-08-13-multitask-threshold-tuning-design.md`

**Note:** `evaluate_multitask` (evaluate.py:485-554) currently ignores
`--sweep_thresholds` — only the classifier/localizer paths use it. The machinery
to reuse already exists: `THRESHOLD_SWEEP = np.arange(0.1, 0.91, 0.05)`
(metrics.py:15) and `find_optimal_threshold` (metrics.py:274).

---

## 21. Threshold tuning results (done)

**Symptom (resolved):** macro F1 stuck at 0.365 with a global 0.5 threshold
despite AUROC 0.75–0.92 — the operating point, not the ranking, was the
bottleneck.

**Fix (implemented, TDD):**
1. `evaluate_multitask` honors `--sweep_thresholds` (per-class `threshold_sweep`
   with `best_f1`/`best_youden`/`f1_at_default`/`sweep` rows; headline
   `macro_f1_at_optimal`).
2. Sweep optima persisted to `<output_dir>/multitask_thresholds.json`
   (`f1_threshold`/`youden_threshold` per class, rounded to 2dp).
3. `MultiTaskClassifier` (registry) resolves per-class thresholds — registry
   `thresholds_path` → sibling `multitask_thresholds.json` next to the model
   file → single-threshold fallback.

**Real run** (best.pt, data/train, 20% stratified val split seed 42, 5857
samples, batch 16): macro F1 **0.365 → 0.467** at per-class optimal thresholds.

| class        | F1@0.5 | F1@opt | f1_thr | youden_thr | AUROC |
|--------------|--------|--------|--------|------------|-------|
| prolongation | 0.422  | 0.422  | 0.50   | 0.40       | 0.820 |
| block        | 0.147  | 0.391  | 0.40   | 0.35       | 0.752 |
| soundrep     | 0.281  | 0.459  | 0.35   | 0.25       | 0.832 |
| wordrep      | 0.249  | 0.337  | 0.40   | 0.35       | 0.756 |
| interjection | 0.725  | 0.725  | 0.50   | 0.40       | 0.916 |
| **macro avg**| **0.365** | **0.467** |      |            |      |

Biggest winners: block +0.24, soundrep +0.18, wordrep +0.09. Interjection and
prolongation peak at 0.50 (already at their optimum).

**Artifacts:** the sweep writes `model/evaluation/reports/multitask_thresholds.json`
as a report artifact; the per-class thresholds are baked into `model/registry.json`
under `classification_multitask.thresholds` (inline, tracked with the registry), so
`MultiTaskClassifier` reads them at load time. The `thresholds_path`/sibling-file
mechanisms remain as fallbacks for unregistered setups.

**Caveats:**
- Thresholds are tuned on the same 20% val split used for the headline numbers
  — optimistic. Held-out validation is next-phase work.
- `f1_at_0_5`/`macro_f1_at_0_5` keys are mislabeled when `--threshold` is
  overridden (they're really "at args.threshold"). Harmless with the default
  0.5; noted for the retrain phase.

**Bonus bug found by the sanity check:** the registry's `_load_multitask_classifier`
used `MultiTaskWav2VecClassifier.from_pretrained` only — crashes with
`KeyError: 'model_name'` on training-format checkpoints (saved by
`save_checkpoint`, no `model_name` key). Fixed by delegating to the robust
`model.evaluation.loader.load_multitask` (same dual-format handling as §18,
which covered the evaluation loader). Test:
`test_multitask_loader_handles_training_format`.

