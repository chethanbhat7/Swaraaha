"""Classification API routes."""

from fastapi import APIRouter, File, UploadFile

from backend.services.classifier import classify_audio_bytes

router = APIRouter()


@router.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    results = classify_audio_bytes(audio_bytes)
    return results
