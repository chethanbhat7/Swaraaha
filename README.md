# Swaraaha

A speech dysfluency (stuttering) detection tool that analyzes audio recordings to classify and localize different types of speech dysfluencies.

## Features

- **Classification** — Identifies the type(s) of dysfluency present in speech:
  - Prolongation
  - Block
  - Sound Repetition
  - Word Repetition
  - Interjection
- **Localization** — Pinpoints exactly where in the audio a dysfluency occurs
- **Dual Interface** — Available as both a web application and a desktop application

## Tech Stack

| Component    | Technology                          |
|--------------|-------------------------------------|
| Frontend     | React, TypeScript, Vite, TailwindCSS |
| Backend      | FastAPI (Python)                    |
| Desktop App  | PySide6 (Python)                    |
| ML Models    | PyTorch, Wav2Vec 2.0, Librosa      |
| Containerization | Docker                        |
| Deployment   | Render                              |

## Project Structure

```
Swaraaha/
├── frontend/       # React web app UI
├── backend/        # FastAPI server and API routes
├── app/            # PySide6 desktop application
├── model/          # ML models, training code, and weights (shared)
└── docs/           # Project documentation
```

---

## Web App Setup

The web app consists of a React frontend and a FastAPI backend.

### Prerequisites

- Node.js (v18+)
- Python 3.11+

### Backend

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r model/requirements.txt
pip install -r backend/requirements.txt

# Run the backend server
uvicorn backend.main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`. A health check endpoint is available at `GET /health`.

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`. API requests are automatically proxied to the backend.

### Using Docker

You can also run the backend using Docker:

```bash
docker compose up --build
```

This builds and starts the backend container on port `8000`. Model weights should be placed in `model/weights/` before building.

### API Endpoints

| Method | Endpoint         | Description                                      |
|--------|------------------|--------------------------------------------------|
| GET    | `/health`        | Health check                                     |
| POST   | `/api/classify`  | Classify dysfluency types in an audio file       |
| POST   | `/api/localize`  | Localize where dysfluencies occur in audio       |
| POST   | `/api/analyze`   | Run both classification and localization at once |

All audio endpoints accept a file upload (`multipart/form-data`).

---

## Desktop App Setup

The desktop app is a standalone PySide6 application.

### Prerequisites

- Python 3.11+

### Running the Desktop App

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r model/requirements.txt
pip install -r app/requirements.txt

# Launch the application
python -m app.main
```

---

## License

This project is private and not currently open for public use.
