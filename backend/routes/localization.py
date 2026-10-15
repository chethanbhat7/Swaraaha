"""Localization API routes."""

from fastapi import APIRouter, File, Form, UploadFile

from backend.services.localizer import localize_audio_bytes
from model import analyze as _analyze

router = APIRouter()


@router.post("/localize")
async def localize_audio(file: UploadFile = File(...), language: str = Form("english")):
    audio_bytes = await file.read()
    return localize_audio_bytes(audio_bytes, language=language)


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), language: str = Form("english")):
    audio_bytes = await file.read()
    results = _analyze(audio_bytes, language=language)

    localization = results.get("localization", {})

    return {
        "classification": results.get("classification", {}),
        "localization": localization,
        "transcription": results.get("transcription", {}),
        "combined": results.get("combined", {}),
    }
