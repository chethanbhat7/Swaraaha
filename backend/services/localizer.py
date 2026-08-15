"""Service layer for the localization pipeline."""

import io
from typing import Optional

import numpy as np

from backend.services.audio_utils import convert_to_wav
from model.data.preprocessing import generate_mel_spectrogram
from model.registry import Localizer

_model: Optional[Localizer] = None


def get_model() -> Localizer:
    global _model
    if _model is None:
        _model = Localizer("cnn")
    return _model


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    import soundfile as sf

    audio_bytes = convert_to_wav(audio_bytes)
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000

    duration_sec = round(len(audio_data) / sr, 3)

    if len(audio_data) > 160000:
        audio_data = audio_data[:160000]
    elif len(audio_data) < 160000:
        audio_data = np.pad(audio_data, (0, 160000 - len(audio_data)))

    # Generate spectrogram: (n_mels, T) — the CNN localizer's predict() input
    spec = generate_mel_spectrogram(audio_data, sr=sr)

    try:
        regions = get_model().predict(spec)
    except Exception as e:
        return {"regions": [], "error": str(e), "duration_sec": duration_sec}

    return {
        "regions": [
            {"start": round(s, 3), "end": round(e, 3), "confidence": round(c, 4)}
            for s, e, c in regions
        ],
        "duration_sec": duration_sec,
    }
