# PROJECT.md — Swaraaha

> Context file for AI coding assistants (Antigravity, Claude Code, Cursor, etc).
> Read this fully before making changes. Update this file whenever architecture,
> conventions, model design, or active work changes. This is the persistent memory
> across sessions — nothing here should be assumed stale just because a session ended.

## 1. What this project is

Swaraaha is a speech dysfluency (stuttering) detection tool. It analyzes a speech
recording and does two things:
1. **Classifies** which type(s) of dysfluency are present
   (`prolongation`, `block`, `soundrep`, `wordrep`, `interjection`).
2. **Localizes** — pinpoints *where in the audio* a dysfluency occurs, not
   just that it occurred somewhere.

It ships as both a **web app** and a **desktop app**.

## 2. Top-level structure

```
Swaraaha/
├── frontend/     # Web app UI (React + Vite + TypeScript)
├── backend/      # Web app API / server (FastAPI)
├── model/        # ML models, training, evaluation — shared source of truth
├── app/          # PySide6 desktop app
└── docs/         # Documentation and design specs
```

- `frontend/` + `backend/` together make up the web app.
- `app/` is a separate, independently-runnable **desktop app** (PySide6).
- `model/` is the single source of truth for model architecture, weights,
  training/eval code, and the model registry. Both `app/` and `backend/`
  load from here via the registry API.

## 3. Tech stack

- **Frontend:** React 19, Vite, TypeScript, Tailwind CSS
- **Backend:** FastAPI (Python 3.11)
- **Desktop app (`app/`):** PySide6
- **ML:** PyTorch, Hugging Face Transformers (Wav2Vec 2.0), librosa
- **Containerization:** Docker
- **Deployment:** Render (backend only)

## 4. Model architecture

There are **two independent model pipelines**, run separately, whose
outputs are shown together rather than merged into one score:

### 4a. Classification pipeline — "what kind of stutter"
- **Five binary classifiers**, one per dysfluency class: `prolongation`,
  `block`, `soundrep`, `wordrep`, `interjection`.
- Each classifier is based on **Wav2Vec 2.0** (`facebook/wav2vec2-base`), fine-tuned per class.
- A **hybrid MLP combiner** (`CombinerMLP`) exists in `model/classification/hybrid.py` but is not used by the registry API — all-5 `analyze()` returns raw per-classifier outputs plus a detected-classes summary.
- All models are loaded via the **model registry** (`model/registry.py` + `model/registry.json`).

### 4b. Localization pipeline — "where in the audio it happened"
- **CNN spectrogram model** — runs convolutional kernels over mel-spectrograms to detect dysfluency regions.
- **Wav2Vec2 temporal attention model** — uses Wav2Vec2 backbone + attention head for frame-level prediction from raw audio.
- Both are loaded via the model registry when trained checkpoints are available.

### How the two pipelines relate
They are **independent**: the classifier pipeline gives "this audio contains
X type of dysfluency," the localization model separately gives "at this
point in the audio." Results are presented together to the user, not
fused into a single model or score.

## 5. Model Registry

The model registry decouples model loading from model training.

**Always use the registry API to access trained models.** Do not instantiate model classes directly or load checkpoints manually. The API accepts audio as a file path, raw bytes, or numpy array and applies preprocessing automatically.

```python
from model import Classifier, Localizer, ModelRegistry

clf = Classifier()                        # all 5 classifiers (raw outputs + summary)
result = clf.analyze("recording.wav")     # path, bytes, or numpy array
# {prolongation: {...}, ..., summary: {detected: [...], primary: "..."}}

clf = Classifier("prolongation")          # single classifier
result = clf.analyze(audio)               # {label, confidence, prob_present, prob_not_present}

result = clf.analyze_raw(audio)           # advanced: same + raw logits
result = clf.analyze(audio, threshold=0.6)  # per-call threshold override

loc = Localizer("cnn")                   # single localizer
m = ModelRegistry()                       # everything at once
m.run_all(audio_tensor)                   # classify + localize in one call
```

- **`model/registry.json`** — lists available model checkpoint paths, grouped by task (classification, localization), plus per-class label `thresholds`.
- **`model/registry.py`** — Python API with `Classifier`, `Localizer`, and `ModelRegistry` classes. Models are lazy-loaded on first call. `Classifier.analyze` handles audio normalization internally (load/resample → clean → pad to 10s).
- The hybrid combiner (learned MLP) is not used by the API — all-5 `analyze()` returns raw per-classifier outputs plus a summary of detected classes.

To change which checkpoint is active, update the path in `registry.json`. No code changes needed. Training does NOT write to the registry — model selection is manual.

**Do not** import `HybridClassifier`, `ProlongationClassifier`, etc. directly — use `Classifier()` instead. This ensures models load from the registry and stay in sync with which checkpoints are active.

## 6. Training

Training uses **parameter-specific fingerprint naming** — all hyperparameters are encoded in the output filename:

```
prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base_best.pt
```

Key features:
- **Checkpoint-based resume** — interrupted training resumes from the last epoch
- **Audio preprocessing cache** — preprocessed audio cached to `data/cache/` for fast re-runs
- **Focal loss** — handles class imbalance without explicit class weights
- **Backbone freezing** — head-only training for first N epochs, then unfreeze with 10× lower LR
- **`torch.compile`** — JIT compilation on CUDA for ~2× speed
- **Gradient accumulation** — effective batch size scaling

All 5 classifiers have been trained. Results (val F1):
- prolongation: 0.5239, block: 0.5288, soundrep: 0.0019, wordrep: 0.0135, interjection: 0.6830

## 7. Conventions

- **Commit messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/):
  ```
  <type>[optional scope]: <description>
  ```
  - **Types:** `fix`, `feat`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`
  - **Scopes:** `(model)`, `(data)`, `(app)`, `(localization)`, `(classification)`, `(training)`, `(eval)`
- Keep `model/` framework-agnostic where possible so both `backend/`
  (FastAPI) and `app/` (PySide6) can import from it without pulling in
  web-only or desktop-only dependencies.
- Model weights (`model/weights/`) are git-ignored — they live locally only.

## 8. Current state

- [x] Wav2Vec 2.0 per-class binary classifiers — `model/classification/`
- [x] Hybrid combiner model (MLP) — `model/classification/hybrid.py`
- [x] CNN spectrogram localization — `model/localization/cnn_spectrogram.py`
- [x] Wav2Vec2 temporal attention localization — `model/localization/wav2vec2_localizer.py`
- [x] Web app (frontend/backend) — React + FastAPI
- [x] Docker setup — `backend.Dockerfile` + `docker-compose.yml`
- [x] Render deployment — `render.yaml`
- [x] Data pipeline — download, merge, prepare (3 datasets: Boli, SEP-28K, UCLASS)
- [x] Training pipeline — all 5 classifiers trained, checkpoint resume, audio cache
- [x] Model registry — JSON + Python API for loading trained models
- [x] PySide6 desktop app — scaffolding complete
- [ ] Localizer training — no trained checkpoints yet
- [ ] Threshold tuning — sweep val set for optimal F1
- [ ] Warm restart LR schedule — extend training beyond 20 epochs
