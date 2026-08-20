"""Service layer for the classification pipeline.

Uses the easy-mode model API (model.analyze).
"""

from model import analyze as _analyze


def classify_audio_bytes(audio_bytes: bytes) -> dict:
    """Classify audio from raw bytes."""
    return _analyze(audio_bytes).get("classification", {})


def classify_audio_file(path: str) -> dict:
    """Classify audio from a file path."""
    return _analyze(path).get("classification", {})
