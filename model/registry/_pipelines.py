"""High-level audio pipeline functions."""

import json

from model.config.defaults import FRAME_DURATION, SAMPLE_RATE

from ._utils import _REGISTRY_PATH

# ---------------------------------------------------------------------------
# Utility functions (use model.classify / model.localize / model.fuse instead)
# ---------------------------------------------------------------------------

def load_synthesis_config() -> dict:
    """Load localization_synthesis config from registry.json."""
    try:
        with open(_REGISTRY_PATH) as f:
            return json.load(f).get("localization_synthesis", {})
    except Exception:
        return {}


def load_audio_16k(audio_bytes: bytes):
    """Load audio bytes -> (16kHz mono float32 array, duration_sec)."""
    import io as _io

    import librosa
    import numpy as _np
    import soundfile as _sf

    from model.data.preprocessing import convert_to_wav

    wav_bytes = convert_to_wav(audio_bytes)
    audio_data, sr = _sf.read(_io.BytesIO(wav_bytes))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio_data = _np.asarray(audio_data, dtype=_np.float32)
    duration_sec = round(len(audio_data) / SAMPLE_RATE, 3)
    return audio_data, duration_sec


def saliency_regions(saliency, class_names, duration_sec: float) -> list:
    """Extract contiguous high-saliency spans per class into regions."""
    import numpy as _np

    cfg = load_synthesis_config()
    min_span_frames = max(1, int(cfg.get("min_span_sec", 0.16) / FRAME_DURATION))
    adapt_k = float(cfg.get("adapt_k", 2.0))
    floor = float(cfg.get("floor", 0.6))
    max_threshold = float(cfg.get("max_threshold", 0.95))

    saliency = _np.asarray(saliency, dtype=float)
    regions = []
    n_frames = saliency.shape[0] if saliency.ndim == 2 else 0
    if n_frames == 0:
        return regions

    for c, name in enumerate(class_names):
        if c >= saliency.shape[1]:
            continue
        col = saliency[:, c]
        threshold = min(max_threshold, max(floor, col.mean() + adapt_k * col.std()))
        on = col >= threshold

        start = None
        for t in range(n_frames + 1):
            if t < n_frames and on[t] and start is None:
                start = t
            elif (t == n_frames or not on[t]) and start is not None:
                end = t
                if end - start >= min_span_frames:
                    seg = col[start:end]
                    regions.append({
                        "start": round(min(start * FRAME_DURATION, duration_sec), 3),
                        "end": round(min(end * FRAME_DURATION, duration_sec), 3),
                        "confidence": round(float(seg.max()), 4),
                        "type": name,
                    })
                start = None

    regions.sort(key=lambda r: r["start"])

    regions.sort(key=lambda r: r["confidence"], reverse=True)
    kept = []
    for region in regions:
        overlaps = any(
            region["start"] < k["end"] and region["end"] > k["start"] for k in kept
        )
        if not overlaps:
            kept.append(region)
    kept.sort(key=lambda r: r["start"])
    return kept


def classify_audio_bytes(audio_bytes: bytes) -> dict:
    """Classify audio from raw bytes. Prefer ``model.classify()`` for new code."""
    from model import classify
    return classify(audio_bytes)


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    """Localize audio from raw bytes. Prefer ``model.localize()`` for new code."""
    from model import localize
    return localize(audio_bytes)


def combine_with_saliency(audio_bytes: bytes, regions: list) -> dict:
    """Fuse regions with saliency. Prefer ``model.fuse()`` for new code."""
    from model import fuse
    return fuse(audio_bytes, regions)
