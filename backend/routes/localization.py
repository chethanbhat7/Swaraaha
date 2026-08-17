"""Localization API routes."""

from fastapi import APIRouter, File, Form, UploadFile

from backend.services.classifier import classify_audio_bytes
from backend.services.fusion import combine_audio_bytes
from backend.services.localizer import localize_audio_bytes
from backend.services.severity import compute_severity
from backend.services.transcriber import transcribe_audio_bytes
router = APIRouter()


@router.post("/localize")
async def localize_audio(file: UploadFile = File(...), language: str = Form("english")):
    audio_bytes = await file.read()
    return localize_audio_bytes(audio_bytes)


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), language: str = Form("english")):
    audio_bytes = await file.read()
    classification = classify_audio_bytes(audio_bytes)
    localization = localize_audio_bytes(audio_bytes)
    transcription = transcribe_audio_bytes(audio_bytes, language=language)
    severity = compute_severity(
        localization["regions"], localization.get("duration_sec") or 0.0
    )
    combined = combine_audio_bytes(audio_bytes, localization.get("regions", []))
    return {
        "classification": classification,
        "localization": localization,
        "transcription": transcription,
        "severity": severity,
        "combined": combined,
    }
