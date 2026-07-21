"""Localization API routes."""

from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/localize")
async def localize_audio(file: UploadFile = File(...)):
    return {"error": "not yet implemented"}


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    return {"error": "not yet implemented"}
