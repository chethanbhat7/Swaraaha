"""Service layer for the localization pipeline."""

import io
from typing import Optional

import numpy as np
import torch

from model.localization.spectrogram import audio_to_mel_spectrogram, load_audio
from model.localization.cnn import SpectrogramCNN
from model.localization.inference import frames_to_regions

_model: Optional[SpectrogramCNN] = None


def get_model() -> SpectrogramCNN:
    global _model
    if _model is None:
        _model = SpectrogramCNN()
    return _model


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    import soundfile as sf

    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000

    if len(audio_data) > 160000:
        audio_data = audio_data[:160000]
    elif len(audio_data) < 160000:
        audio_data = np.pad(audio_data, (0, 160000 - len(audio_data)))

    spec = audio_to_mel_spectrogram(audio_data, sr=sr)
    spec = spec.unsqueeze(0)

    model = get_model()
    probs = model(spec).squeeze(0).tolist()
    regions = frames_to_regions(probs, sr=sr)

    return {"regions": regions}
