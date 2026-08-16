"""Service layer for the localization pipeline.

Dedicated localizer weights (CNN / wav2vec2) are not always available, so
``localize_audio_bytes`` falls back to synthesizing regions from the multitask
classifier's per-frame saliency when the CNN localizer cannot be loaded.
"""

import io
import json
import os
from typing import Optional

import numpy as np

from backend.services.audio_utils import convert_to_wav
from model.config.defaults import DYSFLUENCY_CLASSES, MAX_AUDIO_LENGTH
from model.data.preprocessing import generate_mel_spectrogram
from model.registry import Localizer, MultiTaskClassifier

_model: Optional[Localizer] = None
_multitask: Optional[MultiTaskClassifier] = None

SAMPLE_RATE = 16000
FRAME_DURATION = 320 / SAMPLE_RATE  # 0.02 s per wav2vec2 frame
_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "model",
    "registry.json",
)


def _synthesis_config() -> dict:
    try:
        with open(_REGISTRY_PATH) as f:
            return json.load(f).get("localization_synthesis", {})
    except Exception:
        return {}


def get_model() -> Localizer:
    global _model
    if _model is None:
        _model = Localizer("cnn")
    return _model


def _get_multitask() -> MultiTaskClassifier:
    global _multitask
    if _multitask is None:
        _multitask = MultiTaskClassifier()
    return _multitask


def _saliency_regions(saliency: np.ndarray, class_names, duration_sec: float) -> list:
    """Extract contiguous high-saliency spans per class into regions.

    The multitask heads have per-class biases (prolongation/soundrep are
    elevated across whole clips), so each class uses an adaptive threshold
    ``mean + adapt_k * std`` over the clip, floored by ``floor`` and capped
    by ``max_threshold``. Spans shorter than ``min_span_sec`` are dropped.
    """
    cfg = _synthesis_config()
    min_span_frames = max(1, int(cfg.get("min_span_sec", 0.16) / FRAME_DURATION))
    adapt_k = float(cfg.get("adapt_k", 2.0))
    floor = float(cfg.get("floor", 0.6))
    max_threshold = float(cfg.get("max_threshold", 0.95))

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

    # Drop lower-confidence regions that overlap an already-accepted one so
    # the graph shows one band per time window.
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


def _saliency_fallback(audio_data: np.ndarray, duration_sec: float) -> dict:
    """Synthesize regions from multitask saliency (returns {"error": ...} on failure)."""
    try:
        saliency = _get_multitask().saliency(audio_data).squeeze(0)
        if hasattr(saliency, "cpu"):
            saliency = saliency.cpu()
        saliency = np.asarray(saliency, dtype=float)
        regions = _saliency_regions(saliency, list(DYSFLUENCY_CLASSES), duration_sec)
        return {
            "regions": regions,
            "duration_sec": duration_sec,
            "source": "saliency",
        }
    except Exception as exc:
        return {"regions": [], "error": str(exc), "duration_sec": duration_sec}


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    import soundfile as sf

    audio_bytes = convert_to_wav(audio_bytes)
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != SAMPLE_RATE:
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    duration_sec = round(len(audio_data) / sr, 3)

    if len(audio_data) > MAX_AUDIO_LENGTH:
        audio_data = audio_data[:MAX_AUDIO_LENGTH]
    elif len(audio_data) < MAX_AUDIO_LENGTH:
        audio_data = np.pad(audio_data, (0, MAX_AUDIO_LENGTH - len(audio_data)))

    # Generate spectrogram: (n_mels, T) — the CNN localizer's predict() input
    spec = generate_mel_spectrogram(audio_data, sr=sr)

    try:
        regions = get_model().predict(spec)
    except Exception:
        # Dedicated localizer weights unavailable → use multitask saliency.
        return _saliency_fallback(audio_data, duration_sec)

    return {
        "regions": [
            {"start": round(s, 3), "end": round(e, 3), "confidence": round(c, 4)}
            for s, e, c in regions
        ],
        "duration_sec": duration_sec,
    }
