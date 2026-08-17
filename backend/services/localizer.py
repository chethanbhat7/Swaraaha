"""Service layer for the localization pipeline.

Delegates to model.registry.localize_audio_bytes which handles
preprocessing, localizer inference, and saliency fallback internally.
"""

from typing import Optional

from model.registry import Localizer, localize_audio_bytes as _localize

_model: Optional[Localizer] = None


def get_model() -> Localizer:
    global _model
    if _model is None:
        _model = Localizer("wav2vec2")
    return _model


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    """Localize dysfluency regions from raw audio bytes."""
    return _localize(audio_bytes)
