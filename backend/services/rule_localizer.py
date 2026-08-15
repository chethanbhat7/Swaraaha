"""Rule-based localization: derive dysfluency regions from transcription words.

Falls back to transcription word timestamps when the ML localizer model is
unavailable. A word is a dysfluency event when it repeats the previous word
(word repetition) or contains a repeated syllable/fragment (sound repetition,
e.g. "s-s", "sss").
"""

import re
from typing import Any, Dict, List

_REPEATED_FRAGMENT_RE = re.compile(r"(.)\1{2,}")


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
    """Build dysfluency regions from transcription word timestamps."""
    regions: List[Dict[str, Any]] = []
    prev_text = ""
    prev_start = None
    prev_end = None
    for word in words:
        text = str(word.get("text") or "").strip().lower()
        start, end = word.get("start"), word.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if text == prev_text or _is_repeated_fragment(text):
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
