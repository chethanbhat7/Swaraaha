"""Anonymous correction submissions: pair an audio sample with a free-form comment.

Each submission is written to disk under a unique id so the (audio, text) pair can
be reviewed later for model corrections and improvement. No patient details are
persisted — only the audio bytes, the comment and a little provenance metadata.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "corrections"

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


def _corrections_dir() -> Path:
    configured = os.environ.get("CORRECTIONS_DIR")
    return Path(configured) if configured else DEFAULT_DIR


def _safe_extension(filename: Optional[str]) -> str:
    """Reduce an upload filename to a whitelisted extension (never keep the name)."""
    extension = Path(filename or "").suffix.lower()
    return extension if extension in ALLOWED_EXTENSIONS else ".wav"


def _new_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"corr_{stamp}_{uuid.uuid4().hex[:8]}"


def save_correction(
    audio_bytes: bytes,
    filename: Optional[str] = None,
    comment: str = "",
    language: Optional[str] = None,
) -> dict:
    """Persist an anonymous audio + comment pair and return its unique labels."""
    if not audio_bytes:
        raise ValueError("audio payload is empty")

    base = _corrections_dir()
    base.mkdir(parents=True, exist_ok=True)

    correction_id = _new_id()
    audio_label = f"{correction_id}{_safe_extension(filename)}"
    text_label = f"{correction_id}.txt"
    created_at = datetime.now(timezone.utc).isoformat()

    (base / audio_label).write_bytes(audio_bytes)
    (base / text_label).write_text(comment or "", encoding="utf-8")
    (base / f"{correction_id}.json").write_text(
        json.dumps(
            {
                "id": correction_id,
                "created_at": created_at,
                "language": language,
                "audio_label": audio_label,
                "text_label": text_label,
                "audio_bytes": len(audio_bytes),
                "anonymous": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "id": correction_id,
        "audio_label": audio_label,
        "text_label": text_label,
        "created_at": created_at,
    }
