"""Service layer for the localization pipeline.

Delegates to model.registry LocalizerRunner which handles
preprocessing, localizer inference, and saliency fallback internally.
"""

from typing import Optional

from model.registry import LocalizerRunner

_model: Optional[LocalizerRunner] = None


def get_model() -> LocalizerRunner:
    global _model
    if _model is None:
        _model = LocalizerRunner("wav2vec2")
    return _model


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    """Localize dysfluency regions from raw audio bytes."""
    return get_model().analyze(audio_bytes)
