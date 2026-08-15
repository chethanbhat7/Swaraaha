"""Service layer for the classification pipeline."""

from typing import Optional

from model.registry import MultiTaskClassifier

_clf: Optional[MultiTaskClassifier] = None


def get_model() -> MultiTaskClassifier:
    global _clf
    if _clf is None:
        _clf = MultiTaskClassifier()
    return _clf


def classify_audio_bytes(audio_bytes: bytes) -> dict:
    """Classify audio from raw bytes via the registry API."""
    model = get_model()
    return model.analyze(audio_bytes)


def classify_audio_file(path: str) -> dict:
    """Classify audio from a file path via the registry API."""
    model = get_model()
    return model.analyze(path)
