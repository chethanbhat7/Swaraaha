# Swaraaha

Speech dysfluency (stuttering) detection tool. Analyzes speech recordings to **classify** dysfluency types and **localize** where they occur in the audio.

Dysfluency types detected: `prolongation`, `block`, `soundrep`, `wordrep`, `interjection`.

Ships as both a **web app** and a **desktop app**.

## Architecture

```
Swaraaha/
├── frontend/          # React + Vite + TypeScript web UI
├── backend/           # FastAPI API server
├── model/             # ML models, training, evaluation (shared)
│   ├── classification/  # Wav2Vec 2.0 binary classifiers (one per dysfluency type)
│   ├── localization/    # CNN spectrogram + Wav2Vec2 localization
│   ├── training/        # Training pipelines
│   ├── evaluation/      # Metrics and evaluation scripts
│   ├── data/            # Dataset loading, preprocessing, augmentation
│   ├── config/          # Hyperparameter defaults
│   ├── registry/        # Model registry (runners + registry.json driven loaders)
│   ├── transcription.py # Whisper transcription API (Transcriber)
│   └── registry.json    # Active model paths + thresholds + defaults
├── app/               # PySide6 desktop application
├── docker-compose.yml
├── backend.Dockerfile
└── render.yaml        # Render deployment blueprint
```

**Two pipelines, independent of each other:**

1. **Classification** — five Wav2Vec 2.0 binary classifiers (one per dysfluency type), each answering *is this type present*. Answers *what kind* of stutter.
2. **Localization** — CNN over spectrogram images or Wav2Vec2 temporal attention to pinpoint *where* in the audio a dysfluency occurs.

> **Localizer status.** The first localizer checkpoints were trained on a broken
> label pipeline (SEP-28K timestamps are episode-relative *sample indices*, not
> clip-relative seconds — see `debugging.md`), so they predicted no events and
> returned empty `regions`. The label pipeline has been fixed and both localizers
> have been retrained on the corrected labels — the registered checkpoints now
> produce real regions (wav2vec2 localizer: frame F1 0.773, event detection
> accuracy 0.614 on the test set; see *Model Metrics* below). If a localizer is
> missing from `registry.json`, `model.init()` / `model.analyze()` still return
> `{"error": ...}` — always consume results through the registry API rather than
> hand-rolling a workaround, so new checkpoints are picked up automatically.

Both `frontend/` + `backend/` (web) and `app/` (desktop) load from the shared `model/` directory via the model registry.

## Development Setup

The project uses [uv](https://docs.astral.sh/uv/) for Python toolchain and
dependency management (Python 3.12, pinned in `.python-version`).

Prerequisites: [Install uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Create the .venv and install all dependencies (incl. dev: pytest, ruff)
uv sync

# Verify the setup
uv run pytest
uv run ruff check .
```

Use `uv run <command>` (or the `.venv/bin/*` scripts) instead of activating the
venv, e.g. `uv run uvicorn backend.main:app`. Never use `pip install` / manual
venvs — `pyproject.toml` + `uv.lock` are the single source of truth.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.12, Uvicorn |
| Desktop | PySide6, sounddevice, NumPy |
| ML | PyTorch, Hugging Face Transformers (Wav2Vec 2.0), librosa |
| Container | Docker, docker-compose |
| Deploy | Render (backend only) |

## Accessing Trained Models

**Always use the model registry API** to load and run trained models. Do not instantiate model classes directly or load checkpoints manually. The API accepts audio as a file path, raw bytes, or a numpy array — preprocessing is applied automatically.

```python
import model

# Load default models from registry.json (arm01 single-class classifiers + wav2vec2 localizer)
model.init()
# or explicitly:  model.init(classifier="multitask", localizer="cnn")

# Everything at once — raw audio in, all results out
result = model.analyze("recording.wav", text="the cat sat")
# {classification: {...}, localization: {...}, transcription: {...},
#  combined: {regions: [...], audio_duration, total_stutters}}

# Individual pipelines
clf_result = model.classify("recording.wav", threshold=0.55)
loc_result = model.localize("recording.wav", text="the cat sat", language="en")
tr_result = model.transcribe("recording.wav")
# fuse regions + saliency explicitly
combined  = model.fuse(loc_result, audio, text="the cat sat")

# Per-class classifiers (registry "single" mode, five independent arm01 models)
# clf_result = {prolongation: {label, confidence, prob_present, prob_not_present}, ...}

# CAM saliency (per-frame, per-class softmax on arm01 classifier heads)
from model.registry import ClassifierRunner
clf = ClassifierRunner()
sal = clf.saliency(audio_tensor)  # torch.Tensor shape (1, T, 5)

# Single model runner
from model.registry import ClassifierRunner, LocalizerRunner
clf_prolong = ClassifierRunner("prolongation")
loc = LocalizerRunner()                # type from registry.json (wav2vec2 or cnn)

# Combined output: localizer regions fused with per-class saliency
# each region: {start, end, confidence, classes: {class: {label, confidence,
#   prob_present, prob_not_present}}, primary_type, severity, syllables[]}
# (syllables present only when text is provided)
```

### Combiner & mismatch probe

`combined` (from `model.analyze`) links each localizer region to
per-class dysfluency scores via the classifier's per-frame saliency.
Regions are sorted by start, bounded by audio duration, and optionally snapped
to syllable boundaries. The severity field is a `null` placeholder in v1.

To measure how often high-confidence saliency spans have no overlapping
localizer region (informs a future saliency-synthesis feature):

```bash
python -m model.evaluation.probe_combiner --data_dir data/test --max_length_seconds 3
# {'mean_mismatch': ..., 'clips': ..., 'clips_with_mismatch': ...}
```

The `--data_dir` must be a split dir (e.g. `data/test`) with `audio/` + `labels/`.

The registry (`model/registry.json`) maps task names to checkpoint paths, per-class label thresholds, and a `defaults` block indicating which classifier and localizer to load when `model.init()` is called without arguments. To swap which checkpoint is active, update the path in the JSON file. No code changes needed.

**Do not** import `ProlongationClassifier`, etc. directly — use `ClassifierRunner()` or `model.classify()` instead. This ensures models load from the registry and stay in sync with which checkpoints are active.

**Localizer checkpoints** — CNN and wav2vec2 localizers are registered under `model/registry.json` → `localization`. If a trained localizer is missing, `model.init()` / `model.analyze()` return `{"localization": {"error": ...}}` (or raise `FileNotFoundError`) until a checkpoint is trained and registered. Do not bypass the API with manual preprocessing or ad-hoc model loading; wait for the trained model so everything flows through the registry.

## Dataset Setup

The ML pipelines need audio datasets. Get your Kaggle API key from [kaggle.com/settings](https://www.kaggle.com/settings), then create a `.env` file in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Then run the full setup:

```bash
python -m model.data.setup

# Show available flags
python -m model.data.setup --help

# Force regeneration: re-merge labels and overwrite existing split labels
python -m model.data.setup --force
```

Any extra flags after `--` are forwarded to every step (download, merge,
prepare), e.g. `python -m model.data.setup -- --some-flag value`.

See [`model/data/README.md`](model/data/README.md) for detailed instructions and manual setup options.

## Training

```bash
# Train all 5 classifiers
bash model/training/train_all_classifiers.sh

# Or train one at a time
python -m model.training.train_classifier --class_name prolongation

# Or use the orchestrator
python -m model.training.train
```

- **Localizer pipelines (`loc`, `wav2vec`) resume too**: each run is
  fingerprint-named (`{fp}_checkpoint.pt`, `{fp}_best.pt`, `{fp}_final.pt`,
  `{fp}_log.csv`, `training_curves/{fp}_curves.png`) and a finished run is
  skipped on the next invocation. Use `--clean` to force retraining.

See [`model/training/README.md`](model/training/README.md) for all flags, resume, and tuning options.

## Evaluating Trained Models

Use the evaluation scripts in `model/evaluation/` to benchmark trained checkpoints against the val split — metrics, confusion matrices, threshold sweeps, and JSON/PNG reports.

```bash
# Evaluate a single classifier
python -m model.evaluation.evaluate \
    --model_type classifier --class_name prolongation \
    --model_path model/weights/prolongation_..._best.pt --data_dir data

# Threshold sweep + save misclassified samples
python -m model.evaluation.evaluate --model_type classifier \
    --class_name block --model_path model/weights/block_..._best.pt \
    --data_dir data --sweep_thresholds --save_misclassified
```

See [`model/evaluation/README.md`](model/evaluation/README.md) for full documentation.

## Model Metrics

Evaluation on the held-out test set (3,715 clips). This split is in-distribution (SEP-28K + UCLASS, which share speakers with train) **plus** 53 Project Boli clips. The Boli subset is the only **out-of-distribution (OOD)** hold-out — unseen corpus *and* unseen speakers; Boli never appears in train or val — and is reported separately at the end of this section. The currently deployed models (per `registry.json` defaults) are:

- **Classification:** five independent Wav2Vec2-base binary classifiers (`single` mode — one per dysfluency type, focal loss, naug checkpoint)
- **Localization:** Wav2Vec2-base temporal localizer (naug checkpoint, threshold 0.3)

A shared-head **multitask** classifier is also registered as an alternative; its numbers are listed below for comparison.

### Classification — Five Independent Wav2Vec2 Binary Classifiers (single, default)

**Per-class metrics at default threshold (0.5):**

| Class | Accuracy | Precision | Recall | F1 | AUROC | Support |
|---|---|---|---|---|---|---|
| prolongation | 0.896 | 0.516 | 0.562 | 0.538 | 0.871 | 400 |
| block | 0.831 | 0.425 | 0.380 | 0.401 | 0.761 | 553 |
| soundrep | 0.905 | 0.660 | 0.585 | 0.620 | 0.882 | 494 |
| wordrep | 0.915 | 0.595 | 0.445 | 0.509 | 0.866 | 366 |
| interjection | 0.908 | 0.793 | 0.801 | 0.797 | 0.945 | 834 |
| **Macro avg** | **0.891** | **0.598** | **0.555** | **0.573** | **0.865** | — |

**Per-class metrics at registry (val-optimized) thresholds:**

| Class | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| prolongation | 0.55 | 0.906 | 0.576 | 0.472 | 0.519 |
| block | 0.50 | 0.831 | 0.425 | 0.380 | 0.401 |
| soundrep | 0.45 | 0.898 | 0.621 | 0.609 | 0.615 |
| wordrep | 0.40 | 0.910 | 0.546 | 0.497 | 0.521 |
| interjection | 0.50 | 0.908 | 0.793 | 0.801 | 0.797 |
| **Macro avg** | — | **0.891** | **0.592** | **0.552** | **0.571** |

**Best F1 per class (from threshold sweep):**

| Class | Best Threshold | Best F1 | Best Youden's J |
|---|---|---|---|
| prolongation | 0.50 | 0.538 | 0.574 (t=0.30) |
| block | 0.40 | 0.423 | 0.370 (t=0.30) |
| soundrep | 0.50 | 0.620 | 0.632 (t=0.25) |
| wordrep | 0.35 | 0.522 | 0.573 (t=0.10) |
| interjection | 0.50 | 0.797 | 0.762 (t=0.25) |

### Classification — Wav2Vec2 Multitask (alternative, registered)

**Per-class metrics at default threshold (0.5):**

| Class | Accuracy | Precision | Recall | F1 | AUROC | Support |
|---|---|---|---|---|---|---|
| prolongation | 0.903 | 0.566 | 0.440 | 0.495 | 0.845 | 400 |
| block | 0.863 | 0.586 | 0.264 | 0.364 | 0.767 | 553 |
| soundrep | 0.906 | 0.705 | 0.502 | 0.586 | 0.877 | 494 |
| wordrep | 0.917 | 0.623 | 0.402 | 0.488 | 0.855 | 366 |
| interjection | 0.904 | 0.810 | 0.747 | 0.777 | 0.935 | 834 |
| **Macro avg** | **0.899** | **0.658** | **0.471** | **0.542** | **0.856** | — |

**Per-class metrics at tuned thresholds (val-optimized, from `multitask_thresholds.json`):**

| Class | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| prolongation | 0.55 | 0.906 | 0.603 | 0.367 | 0.457 |
| block | 0.35 | 0.772 | 0.342 | 0.570 | 0.427 |
| soundrep | 0.40 | 0.899 | 0.617 | 0.630 | 0.623 |
| wordrep | 0.35 | 0.891 | 0.455 | 0.552 | 0.499 |
| interjection | 0.45 | 0.897 | 0.770 | 0.775 | 0.772 |
| **Macro avg** | — | **0.873** | **0.557** | **0.579** | **0.556** |

### Localization — Wav2Vec2 Localizer (default)

At the registry's deployed threshold (0.3):

| Metric | Value |
|---|---|
| Frame-level Precision | 0.794 |
| Frame-level Recall | 0.754 |
| Frame-level F1 | 0.773 |
| Frame-level Specificity | 0.833 |
| Event-level Detection Accuracy | 0.614 |
| Event-level Mean IoU | 0.780 |
| True Events | 995 |
| Predicted Events | 2,032 |
| False Alarms | 1,421 |
| False Alarm Rate | 4.81/min |

> **Note:** The retrained wav2vec2 localizer now balances precision and recall
> reasonably well (frame F1 0.773) and detects a majority of true events
> (event accuracy 0.614) — a large improvement over the pre-retrain checkpoint,
> which produced essentially empty results. The CNN localizer is registered as a
> lighter alternative but is much weaker at the event level (detection accuracy
> 0.032 at threshold 0.3) due to over-prediction, so wav2vec2 remains the
> default.

### Out-of-Distribution (OOD) Test — Project Boli

The 53-clip Project Boli subset is the only true **out-of-distribution (OOD)** hold-out: its corpus and speakers appear in neither train nor val, and no threshold was ever tuned on it. It is a generalization sanity check, not a headline claim (small sample; 9–35 positives per class).

**Five single classifiers (default), at registry thresholds:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| prolongation | 1.000 | 0.250 | 0.400 | 16 |
| block | 1.000 | 0.143 | 0.250 | 35 |
| soundrep | 1.000 | 0.147 | 0.256 | 34 |
| wordrep | 0.500 | 0.071 | 0.125 | 14 |
| interjection | 0.600 | 0.333 | 0.429 | 9 |
| **Macro avg** | **0.820** | **0.189** | **0.292** | — |

**Wav2Vec2 multitask (alternative), at tuned thresholds:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| prolongation | 0.500 | 0.125 | 0.200 | 16 |
| block | 0.750 | 0.086 | 0.154 | 35 |
| soundrep | 1.000 | 0.147 | 0.256 | 34 |
| wordrep | 0.250 | 0.143 | 0.182 | 14 |
| interjection | 0.500 | 0.333 | 0.400 | 9 |
| **Macro avg** | **0.600** | **0.167** | **0.238** | — |

**Wav2Vec2 localizer (default), threshold 0.3:** frame F1 0.417 (P 0.327 / R 0.576 / Spec 0.834); event detection accuracy 0.125, mean IoU 0.638, 18 false alarms (4.27/min).

> All models degrade sharply OOD (single macro F1 0.292 vs 0.573 in-distribution): recall collapses far more than precision, so the models conservatively under-detect dysfluencies from unseen speakers/corpus. This is the honest cross-corpus generalization number.

### Key Observations

- **Interjection** remains the strongest class across both architectures (F1 0.797 single / 0.777 multitask, AUROC >0.93) — filler words are the easiest to detect.
- **Block** remains the weakest class (F1 0.401 single / 0.364 multitask) — silent pauses are hard to distinguish from normal speech pauses.
- The **single-classifier** setup (five independent models) slightly edges out the **multitask** model on macro F1 (0.573 vs 0.542 at threshold 0.5), likely because each binary head can specialise without sharing capacity.
- **Accuracy is high across all classes (0.83–0.92)** due to class imbalance — most clips do not contain a given dysfluency type, so predicting "not present" is correct most of the time. F1 is the more meaningful metric for detection performance.
- **Localization** now works meaningfully (frame F1 0.773, event accuracy 0.614) after the label-pipeline fix and retraining — the combiner still helps by fusing localizer regions with the classifier's per-frame saliency, but the localizer is no longer a near-empty signal.
- **Out-of-distribution generalization is weak** (single macro F1 0.292, multitask 0.238 on the unseen-speaker/corpus Boli set) — the in-distribution test numbers are optimistic because SEP-28K/UCLASS test clips share speakers with train. The single-classifier setup again edges out multitask (0.292 vs 0.238).

## Running the Web App

### Backend

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

API runs at `http://localhost:8000`. Health check: `GET /health`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dev server runs at `http://localhost:5173`.

### With Docker

```bash
docker-compose up --build
```

Backend available at `http://localhost:8000`.

## Running the Desktop App

```bash
uv run python -m app.main
```
