"""Localization API routes."""

from fastapi import APIRouter, File, Form, UploadFile

from backend.services.analysis import finalize_localization
from backend.services.classifier import classify_audio_bytes
from backend.services.localizer import localize_audio_bytes
from backend.services.severity import compute_severity
from backend.services.transcriber import transcribe_audio_bytes

router = APIRouter()


@router.post("/localize")
async def localize_audio(file: UploadFile = File(...), language: str = Form("english")):
    audio_bytes = await file.read()
    localization = localize_audio_bytes(audio_bytes)
    if not localization.get("regions"):
        transcription = transcribe_audio_bytes(audio_bytes, language=language)
        localization = finalize_localization(localization, transcription)
    return localization


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), language: str = Form("english")):
    audio_bytes = await file.read()
    classification = classify_audio_bytes(audio_bytes)
    localization = localize_audio_bytes(audio_bytes)
    transcription = transcribe_audio_bytes(audio_bytes, language=language)
    localization = finalize_localization(localization, transcription)
    severity = compute_severity(localization["regions"], localization.get("duration_sec") or 0.0)
    return {
        "classification": classification,
        "localization": localization,
        "transcription": transcription,
        "severity": severity,
    }
