"""Fusion service: combine localizer regions with classifier saliency."""

import io

import numpy as np

from backend.services import classifier
from backend.services.audio_utils import convert_to_wav
from model.registry import DYSFLUENCY_CLASSES, MAX_AUDIO_LENGTH, combine_regions

SAMPLE_RATE = 16000


def _load_16k_mono(audio_bytes: bytes) -> np.ndarray:
    """Decode raw audio bytes to 16kHz mono float32, capped at 10s."""
    import soundfile as sf

    wav_bytes = convert_to_wav(audio_bytes)
    audio_data, sr = sf.read(io.BytesIO(wav_bytes))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.asarray(audio_data, dtype=np.float32)[:MAX_AUDIO_LENGTH]


def combine_audio_bytes(audio_bytes: bytes, regions: list) -> dict:
    """Fuse localizer ``regions`` with classifier saliency.

    Returns the combiner output, or ``{"error": str}`` on any failure so the
    API can degrade gracefully.
    """
    try:
        audio = _load_16k_mono(audio_bytes)
        model = classifier.get_model()
        saliency = model.saliency(audio).squeeze(0)
        if hasattr(saliency, "cpu"):
            saliency = saliency.cpu()
        saliency = np.asarray(saliency, dtype=float)
        return combine_regions(
            regions,
            saliency,
            class_names=list(DYSFLUENCY_CLASSES),
            thresholds=getattr(model, "_thresholds", {}) or None,
            audio_duration=len(audio) / SAMPLE_RATE,
        )
    except Exception as exc:
        return {"error": str(exc)}
