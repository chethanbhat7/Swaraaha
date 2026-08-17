"""Route-level tests for /analyze (combined) and /report (shared builder)."""

import asyncio
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException, UploadFile

from backend.routes.localization import analyze_audio
from backend.routes.report import generate_report

PDF_PAYLOAD = {
    "patient": {"name": "Aarav Sharma", "phone": "999-000-1111"},
    "audio": {"filename": "sample.wav", "size": "2.1 MB", "duration": "4.00 s"},
    "date": "2026-01-02",
    "classification": {"block": {"label": 1, "confidence": 0.87}},
    "combined": {
        "regions": [
            {
                "start": 0.5,
                "end": 1.2,
                "confidence": 0.87,
                "primary_type": "block",
                "classes": {},
                "syllables": [],
            }
        ],
        "audio_duration": 4.0,
        "total_stutters": 1,
    },
    "transcription": {"text": "hello world"},
}


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


def test_analyze_route_includes_combined(monkeypatch):
    monkeypatch.setattr(
        "backend.routes.localization.classify_audio_bytes",
        lambda b: {"prolongation": {"label": 1, "confidence": 0.9}},
    )
    monkeypatch.setattr(
        "backend.routes.localization.localize_audio_bytes",
        lambda b: {"regions": [{"start": 0.0, "end": 0.5, "confidence": 0.9}]},
    )
    monkeypatch.setattr(
        "backend.routes.localization.transcribe_audio_bytes",
        lambda b, language="english": {"text": "hi", "language": language, "chunks": []},
    )
    monkeypatch.setattr(
        "backend.routes.localization.combine_audio_bytes",
        lambda b, regions: {"regions": regions, "audio_duration": 0.5, "total_stutters": 1},
    )

    async def _call():
        return await analyze_audio(
            file=UploadFile(file=io.BytesIO(_wav_bytes()), filename="t.wav")
        )

    result = asyncio.run(_call())
    assert set(result.keys()) == {"classification", "localization", "transcription", "combined", "severity"}
    assert result["combined"]["total_stutters"] == 1


def test_report_route_missing_keys_422():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(generate_report({}))
    assert exc.value.status_code == 422


def test_report_route_returns_pdf():
    resp = asyncio.run(generate_report(PDF_PAYLOAD))
    assert resp.media_type == "application/pdf"
    assert resp.body[:4] == b"%PDF"
    assert "swaraaha-report-2026-01-02.pdf" in resp.headers["Content-Disposition"]
