"""Localization API routes."""

from fastapi import APIRouter, File, UploadFile

from backend.services.classifier import classify_audio_bytes
from backend.services.localizer import localize_audio_bytes

router = APIRouter()


@router.post("/localize")
async def localize_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    results = localize_audio_bytes(audio_bytes)
    return results


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    classification = classify_audio_bytes(audio_bytes)
    localization = localize_audio_bytes(audio_bytes)
    return {
        "classification": classification,
        "localization": localization,
    }
