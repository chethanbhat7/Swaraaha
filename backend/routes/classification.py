"""Classification API routes."""

from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    return {"error": "not yet implemented"}
