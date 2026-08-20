"""Service layer for the classification pipeline."""

from typing import Optional

from model.registry import MultiTaskRunner

_clf: Optional[MultiTaskRunner] = None


def get_model() -> MultiTaskRunner:
    global _clf
    if _clf is None:
        _clf = MultiTaskRunner()
    return _clf


def classify_audio_bytes(audio_bytes: bytes) -> dict:
    """Classify audio from raw bytes via the registry API."""
    model = get_model()
    return model.analyze(audio_bytes)


def classify_audio_file(path: str) -> dict:
    """Classify audio from a file path via the registry API."""
    model = get_model()
    return model.analyze(path)
