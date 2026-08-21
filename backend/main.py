"""Swaraaha FastAPI Backend — serves classification and localization APIs."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(application: FastAPI):
    from backend.routes import classification, localization, report
    application.include_router(classification.router, prefix="/api")
    application.include_router(localization.router, prefix="/api")
    application.include_router(report.router, prefix="/api")
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


@app.get("/health")
def health_check():
    return {"status": "ok"}
