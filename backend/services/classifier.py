"""Service layer for the classification pipeline."""

import logging
from typing import Optional

from model.registry import Classifier, MultiTaskClassifier

logger = logging.getLogger(__name__)

_clf: Optional[object] = None


def get_model() -> object:
    """Return the classification model, preferring the multitask classifier.

    Falls back to the per-class ``Classifier`` when the multitask weights
    are absent from disk (e.g. they were never trained/downloaded), so the
    webapp stays functional with whatever checkpoints are present.
    """
    global _clf
    if _clf is None:
        _clf = MultiTaskClassifier()
    return _clf


def _fallback_model() -> object:
    """Build (or return) the per-class classifier used when multitask is unavailable."""
    global _clf
    if not isinstance(_clf, Classifier):
        _clf = Classifier()
    return _clf


def classify_audio_bytes(audio_bytes: bytes) -> dict:
    """Classify audio from raw bytes via the registry API.

    Prefers the multitask classifier; if its weights are missing (deferred
    FileNotFoundError at analyze time), retries once with the per-class
    ``Classifier`` and keeps that for subsequent calls.
    """
    try:
        return get_model().analyze(audio_bytes)
    except FileNotFoundError:
        logger.warning(
            "MultiTaskClassifier unavailable (weights missing); "
            "falling back to per-class Classifier",
            exc_info=True,
        )
        return _fallback_model().analyze(audio_bytes)


def classify_audio_file(path: str) -> dict:
    """Classify audio from a file path via the registry API."""
    model = get_model()
    return model.analyze(path)
