"""Service layer for the localization pipeline.

Uses the easy-mode model API (model.analyze).
"""

from model import analyze as _analyze


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    """Localize dysfluency regions from raw audio bytes."""
    return _analyze(audio_bytes).get("localization", {})
