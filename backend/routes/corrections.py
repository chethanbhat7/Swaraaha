"""Correction submission route — stores anonymous audio + comment pairs."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.services.corrections import save_correction

router = APIRouter()

MAX_COMMENT_CHARS = 2000


@router.post("/corrections")
async def submit_correction(
    file: UploadFile = File(...),
    comment: str = Form(""),
    language: str = Form(""),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file")

    try:
        return save_correction(
            audio_bytes,
            filename=file.filename,
            comment=(comment or "").strip()[:MAX_COMMENT_CHARS],
            language=language or None,
        )
    except OSError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to store correction") from exc
