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
│   ├── classification/  # Wav2Vec 2.0 binary classifiers + hybrid combiner
│   ├── localization/    # CNN spectrogram-image localization
│   ├── training/        # Training pipelines
│   └── evaluation/      # Metrics and evaluation scripts
├── app/               # PySide6 desktop application
├── docker-compose.yml
├── backend.Dockerfile
└── render.yaml        # Render deployment blueprint
```

**Two pipelines, independent of each other:**

1. **Classification** — five Wav2Vec 2.0 binary classifiers (one per dysfluency type) combined via a hybrid MLP model. Answers *what kind* of stutter.
2. **Localization** — CNN over spectrogram images to pinpoint *where* in the audio a dysfluency occurs.

Both `frontend/` + `backend/` (web) and `app/` (desktop) load from the shared `model/` directory.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11, Uvicorn |
| Desktop | PySide6, sounddevice, NumPy |
| ML | PyTorch, Hugging Face Transformers (Wav2Vec 2.0), librosa |
| Container | Docker, docker-compose |
| Deploy | Render (backend only) |

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
