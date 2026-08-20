"""Service layer for the localization pipeline.

Uses the targeted model.localize() API.
"""

from model import localize as _localize


def localize_audio_bytes(audio_bytes: bytes, language: str = "english") -> dict:
    """Localize dysfluency regions from raw audio bytes."""
    return _localize(audio_bytes, language=language)
