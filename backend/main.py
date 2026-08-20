"""Swaraaha FastAPI backend and single-container frontend host."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.routes import classification, localization, report

app = FastAPI(title="Swaraaha API", version="0.1.0")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
FRONTEND_DIST_DIR = Path(os.environ.get("FRONTEND_DIST_DIR", "/app/frontend/dist"))
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classification.router, prefix="/api")
app.include_router(localization.router, prefix="/api")
app.include_router(report.router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/{path:path}", include_in_schema=False)
def serve_frontend(path: str):
    if path == "api" or path.startswith("api/"):
        raise HTTPException(status_code=404)

    candidate = FRONTEND_DIST_DIR / path
    if candidate.is_file():
        return FileResponse(candidate)

    if not FRONTEND_INDEX_FILE.is_file():
        raise HTTPException(status_code=404)

    return FileResponse(FRONTEND_INDEX_FILE)
