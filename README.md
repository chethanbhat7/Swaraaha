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
│   ├── registry.py      # Model registry API (Classifier, Localizer, ModelRegistry)
│   ├── transcription.py # Whisper transcription API (Transcriber)
│   └── registry.json    # Active model paths
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
> clip-relative seconds — see `debugging.md`), so they predict no events and
> currently return empty `regions`. The label pipeline has been fixed and the
> localizers are being retrained. Until new checkpoints land, `Localizer()`
> / `ModelRegistry.run_all()` may return empty results or `{"error": ...}`.
> Do **not** hand-roll your own localization/preprocessing pipeline or work
> around the registry API to "make it work" in the meantime — use the API and
> handle empty/error results. When the retrained checkpoints land they will be
> wired through `registry.json` automatically; no consumer changes needed.

Both `frontend/` + `backend/` (web) and `app/` (desktop) load from the shared `model/` directory via the model registry.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11, Uvicorn |
| Desktop | PySide6, sounddevice, NumPy |
| ML | PyTorch, Hugging Face Transformers (Wav2Vec 2.0), librosa |
| Container | Docker, docker-compose |
| Deploy | Render (backend only) |

## Accessing Trained Models

**Always use the model registry API** to load and run trained models. Do not instantiate model classes directly or load checkpoints manually. The API accepts audio as a file path, raw bytes, or a numpy array — preprocessing is applied automatically.

```python
from model import Classifier, Localizer, Transcriber, ModelRegistry

# All classifiers (raw per-classifier outputs + summary)
clf = Classifier()
result = clf.analyze("recording.wav")        # path, bytes, or numpy array
# {prolongation: {...}, ..., summary: {detected: [...], primary: "..."}}

# Single classifier
clf = Classifier("prolongation")
result = clf.analyze(audio)                  # {label, confidence, prob_present, prob_not_present}

# Advanced: adds raw logits per class
result = clf.analyze_raw(audio)

# Per-call threshold override (defaults come from model/registry.json)
result = clf.analyze(audio, threshold=0.6)

# Localization: regions always; words/syllables when text is provided
loc = Localizer()                            # type(s) come from registry.json
loc.analyze("recording.wav")                          # regions only
loc.analyze("recording.wav", text="the cat sat", language="en")  # + words + syllables

# Transcription: word-level timestamps + stutter flagging
tr = Transcriber()
tr.transcribe("recording.wav")               # {text, words, duration_sec}

# Everything at once — raw audio in, all results out
m = ModelRegistry()
all_results = m.run_all("recording.wav", text="the cat sat")
# {classification: {...}, cnn_multitask: {...}, multitask: {...},
#  localization: {...}, transcription: {...},
#  combined: {regions: [...], audio_duration, total_stutters}}

# Combined output: localizer regions fused with per-class saliency
# each region: {start, end, confidence, classes: {class: {label, confidence,
#   prob_present, prob_not_present}}, primary_type, severity, syllables[]}
# (syllables present only when text is provided)
```

### Combiner & mismatch probe

`combined` (from `ModelRegistry.run_all`) links each localizer region to
per-class dysfluency scores via the multitask classifier's per-frame saliency.
Regions are sorted by start, bounded by audio duration, and optionally snapped
to syllable boundaries. The severity field is a `null` placeholder in v1.

To measure how often high-confidence saliency spans have no overlapping
localizer region (informs a future saliency-synthesis feature):

```bash
python -m model.evaluation.probe_combiner --data_dir data/test --max_length_seconds 3
# {'mean_mismatch': ..., 'clips': ..., 'clips_with_mismatch': ...}
```

The `--data_dir` must be a split dir (e.g. `data/test`) with `audio/` + `labels/`.

The registry (`model/registry.json`) maps task names to checkpoint paths and per-class label thresholds. To swap which checkpoint is active, update the path in the JSON file. No code changes needed.

**Do not** import `ProlongationClassifier`, etc. directly — use `Classifier()` instead. This ensures models load from the registry and stay in sync with which checkpoints are active.

**Localizer checkpoints** — CNN and wav2vec2 localizers are registered under `model/registry.json` → `localization`. If a trained localizer is missing, `Localizer()` and `ModelRegistry.run_all()` return `{"localization": {"error": ...}}` (or raise `FileNotFoundError`) until a checkpoint is trained and registered. Do not bypass the API with manual preprocessing or ad-hoc model loading; wait for the trained model so everything flows through the registry.

## Dataset Setup

The ML pipelines need audio datasets. Get your Kaggle API key from [kaggle.com/settings](https://www.kaggle.com/settings), then create a `.env` file in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Then run the full setup:

```bash
pip install -r model/requirements.txt
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

Evaluation on the held-out test set (3,715 clips). The currently deployed models are:

- **Classification:** Wav2Vec2-base multitask classifier (5 shared heads, frozen 3 epochs)
- **Localization:** Wav2Vec2-base temporal localizer

### Classification — Wav2Vec2 Multitask (ml3)

**Per-class metrics at default threshold (0.5):**

| Class | Accuracy | Precision | Recall | F1 | AUROC | Support |
|---|---|---|---|---|---|---|
| prolongation | 0.902 | 0.560 | 0.417 | 0.478 | 0.844 | 400 |
| block | 0.861 | 0.638 | 0.150 | 0.243 | 0.735 | 553 |
| soundrep | 0.882 | 0.564 | 0.490 | 0.524 | 0.860 | 494 |
| wordrep | 0.912 | 0.581 | 0.383 | 0.461 | 0.830 | 366 |
| interjection | 0.896 | 0.836 | 0.665 | 0.741 | 0.931 | 834 |
| **Macro avg** | **0.891** | **0.636** | **0.421** | **0.490** | **0.860** | — |

**Per-class metrics at tuned thresholds (val-optimized):**

| Class | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| prolongation | 0.45 | 0.870 | 0.467 | 0.500 | 0.483 |
| block | 0.40 | 0.822 | 0.387 | 0.380 | 0.383 |
| soundrep | 0.40 | 0.824 | 0.460 | 0.658 | 0.541 |
| wordrep | 0.40 | 0.835 | 0.413 | 0.492 | 0.449 |
| interjection | 0.40 | 0.844 | 0.731 | 0.772 | 0.751 |
| **Macro avg** | — | **0.839** | **0.492** | **0.560** | **0.522** |

**Best F1 per class (from threshold sweep):**

| Class | Best Threshold | Best F1 | Best Youden's J |
|---|---|---|---|
| prolongation | 0.45 | 0.483 | 0.514 (t=0.35) |
| block | 0.35 | 0.401 | 0.345 (t=0.35) |
| soundrep | 0.45 | 0.551 | 0.571 (t=0.30) |
| wordrep | 0.45 | 0.475 | 0.503 (t=0.30) |
| interjection | 0.45 | 0.755 | 0.712 (t=0.30) |

### Localization — Wav2Vec2 Localizer (ml3)

| Metric | Value |
|---|---|
| Frame-level Precision | 0.676 |
| Frame-level Recall | 0.065 |
| Frame-level F1 | 0.119 |
| Frame-level Specificity | 0.973 |
| Event-level Detection Accuracy | 0.210 |
| Event-level Mean IoU | 0.751 |
| True Events | 995 |
| Predicted Events | 2,852 |
| False Alarms | 2,643 |
| False Alarm Rate | 8.95/min |

> **Note:** The localizer has high precision (0.676) but low recall (0.065) — it
> is conservative and misses many dysfluency events. When it does predict, the
> regions are reasonably accurate (mean IoU = 0.751). The combiner compensates
> by fusing localizer regions with the classifier's per-frame saliency.

### Key Observations

- **Interjection** is the strongest class (F1=0.741, AUROC=0.931) — the model detects filler words well.
- **Block** is the weakest class (F1=0.243) — silent pauses are hard to distinguish from normal speech pauses.
- **Accuracy is high across all classes (0.82–0.91)** due to class imbalance — most clips do not contain a given dysfluency type, so predicting "not present" is correct most of the time. F1 is the more meaningful metric for detection performance.
- **Localization** needs improvement — the low recall means many dysfluencies are missed at the temporal level.

## Running the Web App

### Backend

Note: run the backend from the root `.venv` (Python 3.10, has all ML deps).
Do not use `backend/.venv` — it is incomplete (missing numpy/torch).

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
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
cd app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ../model/requirements.txt
python -m app.main
```
