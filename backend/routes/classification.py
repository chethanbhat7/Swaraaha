"""Classification API routes."""

from fastapi import APIRouter, File, Form, UploadFile

from backend.services.classifier import classify_audio_bytes
from backend.services.transcriber import transcribe_audio_bytes

router = APIRouter()


@router.post("/classify")
async def classify_audio(file: UploadFile = File(...), language: str = Form("english")):
    audio_bytes = await file.read()
    classification = classify_audio_bytes(audio_bytes)
    transcription = transcribe_audio_bytes(audio_bytes, language=language)
    return {
        "classification": classification,
        "transcription": transcription,
    }
