"""Fusion service: combine localizer regions with classifier saliency.

Uses model.fuse() for the core operation.
"""

from model import fuse as _fuse


def combine_audio_bytes(audio_bytes: bytes, regions: list) -> dict:
    """Fuse localizer regions with classifier saliency."""
    return _fuse(audio_bytes, regions)
