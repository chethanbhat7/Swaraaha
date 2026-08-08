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
│   ├── localization/    # CNN spectrogram + Wav2Vec2 localization (yet to be trained)
│   ├── training/        # Training pipelines
│   ├── evaluation/      # Metrics and evaluation scripts
│   ├── data/            # Dataset loading, preprocessing, augmentation
│   ├── config/          # Hyperparameter defaults
│   ├── registry.py      # Model registry API (Classifier, Localizer, ModelRegistry)
│   └── registry.json    # Active model paths
├── app/               # PySide6 desktop application
├── docker-compose.yml
├── backend.Dockerfile
└── render.yaml        # Render deployment blueprint
```

**Two pipelines, independent of each other:**

1. **Classification** — five Wav2Vec 2.0 binary classifiers (one per dysfluency type), each answering *is this type present*. Answers *what kind* of stutter.
2. **Localization** — CNN over spectrogram images or Wav2Vec2 temporal attention to pinpoint *where* in the audio a dysfluency occurs. (Localizer models are not trained yet — see `registry.json`.)

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
from model import Classifier, Localizer, ModelRegistry

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

# Everything at once
m = ModelRegistry()
all_results = m.run_all(audio_tensor)        # classify + localize
```

The registry (`model/registry.json`) maps task names to checkpoint paths and per-class label thresholds. To swap which checkpoint is active, update the path in the JSON file. No code changes needed.

**Do not** import `ProlongationClassifier`, etc. directly — use `Classifier()` instead. This ensures models load from the registry and stay in sync with which checkpoints are active.

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
```

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

## Running the Web App

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ../model/requirements.txt
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
