"""Assemble the localization payload, falling back to transcription-based regions."""

from backend.services.rule_localizer import regions_from_words


def finalize_localization(localization: dict, transcription: dict) -> dict:
    """Return a localization dict with populated regions.

    Uses the ML localizer output when regions are present; otherwise derives
    regions from transcription word timestamps (rule-based fallback).
    """
    regions = list(localization.get("regions") or [])
    if regions:
        source = "model"
    else:
        words = [chunk for chunk in (transcription.get("chunks") or []) if isinstance(chunk, dict)]
        regions = regions_from_words(words)
        source = "rule-based"

    return {
        "regions": regions,
        "source": source,
        "error": None,
        "duration_sec": localization.get("duration_sec", 0.0),
    }
