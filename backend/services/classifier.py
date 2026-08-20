"""Service layer for the classification pipeline.

Uses the targeted model.classify() API.
"""

from model import classify as _classify


def classify_audio_bytes(audio_bytes: bytes) -> dict:
    """Classify audio from raw bytes."""
    return _classify(audio_bytes)


def classify_audio_file(path: str) -> dict:
    """Classify audio from a file path."""
    return _classify(path)
