"""Swaraaha FastAPI Backend — serves classification and localization APIs."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import classification, localization, report

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML models at server boot."""
    logger.info("Loading ML models...")
    try:
        from model import init as model_init
        model_init()
        logger.info("ML models loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load ML models: %s", exc)
    yield


app = FastAPI(title="Swaraaha API", version="0.1.0", lifespan=lifespan)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

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
