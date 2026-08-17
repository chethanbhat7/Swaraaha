"""Fusion service: combine localizer regions with classifier saliency.

Delegates to model.registry.combine_with_saliency which handles
audio loading, saliency computation, and region fusion internally.
"""

from model.registry import combine_with_saliency as _combine


def combine_audio_bytes(audio_bytes: bytes, regions: list) -> dict:
    """Fuse localizer regions with classifier saliency."""
    return _combine(audio_bytes, regions)
