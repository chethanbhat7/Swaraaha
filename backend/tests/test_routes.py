"""Route-level tests for /analyze (combined) and /report (shared builder)."""

import asyncio
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import backend.main as backend_main
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
    fake_results = {
        "classification": {"prolongation": {"label": 1, "confidence": 0.9}},
        "localization": {"regions": [{"start": 0.0, "end": 0.5, "confidence": 0.9}]},
        "transcription": {"text": "hi", "language": "english", "chunks": []},
        "combined": {"regions": [{"start": 0.0, "end": 0.5, "confidence": 0.9}], "audio_duration": 0.5, "total_stutters": 1},
    }
    monkeypatch.setattr(
        "backend.routes.localization._analyze",
        lambda audio_bytes, language="english": fake_results,
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


def test_frontend_fallback_serves_index_and_skips_api(monkeypatch):
    dist_dir = Path(tempfile.mkdtemp())
    (dist_dir / "index.html").write_text("<!doctype html><title>Swaraaha</title>", encoding="utf-8")
    (dist_dir / "assets").mkdir()
    (dist_dir / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")

    monkeypatch.setattr(backend_main, "FRONTEND_DIST_DIR", dist_dir)
    monkeypatch.setattr(backend_main, "FRONTEND_INDEX_FILE", dist_dir / "index.html")

    client = TestClient(backend_main.app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers["content-type"]
    assert "Swaraaha" in root_resp.text

    asset_resp = client.get("/assets/app.js")
    assert asset_resp.status_code == 200
    assert asset_resp.text == "console.log('ok')"

    api_resp = client.get("/api/does-not-exist")
    assert api_resp.status_code == 404
