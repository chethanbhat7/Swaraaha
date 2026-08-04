"""Service layer for the classification pipeline."""

import io
from typing import Optional

import numpy as np
import torch

from model.registry import Classifier
from backend.services.audio_utils import convert_to_wav

_clf: Optional[Classifier] = None


def get_model() -> Classifier:
    global _clf
    if _clf is None:
        _clf = Classifier()
    return _clf


def classify_audio_bytes(audio_bytes: bytes) -> dict:
    import soundfile as sf

    audio_bytes = convert_to_wav(audio_bytes)
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000

    if len(audio_data) > 160000:
        audio_data = audio_data[:160000]
    elif len(audio_data) < 160000:
        audio_data = np.pad(audio_data, (0, 160000 - len(audio_data)))

    tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    model = get_model()
    results = model.predict(tensor)

    return {
        name: {"label": label, "confidence": round(conf, 4)}
        for name, (label, conf) in results.items()
    }
