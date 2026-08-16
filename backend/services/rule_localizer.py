"""Rule-based localization: derive dysfluency regions from transcription words.

Falls back to transcription word timestamps when the ML localizer model is
unavailable. A word is a dysfluency event when it repeats the previous word
(word repetition) or contains a repeated syllable/fragment (sound repetition,
e.g. "s-s", "sss").
"""

import re
from typing import Any, Dict, List

_REPEATED_FRAGMENT_RE = re.compile(r"(.)\1{2,}")
_WORD_SPLIT_RE = re.compile(r"\s+")
_PUNCT_TRIM = ".,?!;:'\"()[]{} "


def _compare_text(word: str) -> str:
    """Lowercase and strip surrounding punctuation for equality checks."""
    return word.lower().strip(_PUNCT_TRIM)


def _expand_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split a (possibly multi-word) transcription chunk into per-word entries.

    Whisper returns segment-level chunks whose ``text`` may contain several
    words (e.g. ``"I I like tea"``). Distribute the chunk's time span evenly
    across its words so repetitions inside a segment are still detected.
    """
    text = str(chunk.get("text") or "").strip()
    start, end = chunk.get("start"), chunk.get("end")
    if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return [chunk]
    tokens = [t for t in _WORD_SPLIT_RE.split(text) if t]
    if len(tokens) <= 1:
        return [chunk]
    span = float(end) - float(start)
    step = span / len(tokens)
    return [
        {
            "text": token,
            "start": round(float(start) + i * step, 3),
            "end": round(float(start) + (i + 1) * step, 3),
        }
        for i, token in enumerate(tokens)
    ]


def _is_repeated_fragment(word: str) -> bool:
    """Detect syllable/letter repetitions inside a single token."""
    if not word:
        return False
    if "-" in word:
        parts = [p for p in word.split("-") if p]
        if len(parts) >= 2 and len(set(parts)) == 1:
            return True
    return bool(_REPEATED_FRAGMENT_RE.search(word))


def _merge_regions(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge regions that overlap or are separated by at most 0.05s."""
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: r["start"])
    merged = [dict(ordered[0])]
    for region in ordered[1:]:
        last = merged[-1]
        if region["start"] <= last["end"] + 0.05:
            last["end"] = max(last["end"], region["end"])
        else:
            merged.append(dict(region))
    for region in merged:
        region["start"] = round(region["start"], 3)
        region["end"] = round(region["end"], 3)
    return merged


def regions_from_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build dysfluency regions from transcription word/chunk timestamps."""
    regions: List[Dict[str, Any]] = []
    prev_text = ""
    prev_start = None
    prev_end = None
    for word in words:
        for entry in _expand_chunk(word):
            raw_text = str(entry.get("text") or "").strip()
            text = _compare_text(raw_text)
            start, end = entry.get("start"), entry.get("end")
            if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            if text == prev_text or _is_repeated_fragment(raw_text.lower()):
                region_start = float(start)
                if text == prev_text and prev_start is not None and float(start) <= float(prev_end) + 0.05:
                    region_start = float(prev_start)
                regions.append({
                    "start": region_start,
                    "end": float(end),
                    "type": "wordrep" if text == prev_text else "soundrep",
                    "confidence": 1.0,
                })
            prev_text = text
            prev_start = start
            prev_end = end
    return _merge_regions(regions)
