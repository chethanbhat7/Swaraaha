"""Service layer for the localization pipeline."""

import io
from typing import Optional

import numpy as np
import torch

from model.data.preprocessing import generate_mel_spectrogram, load_audio
from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer
from backend.services.audio_utils import convert_to_wav

_model: Optional[CNNSpectrogramLocalizer] = None


def get_model() -> CNNSpectrogramLocalizer:
    global _model
    if _model is None:
        _model = CNNSpectrogramLocalizer()
    return _model


def localize_audio_bytes(audio_bytes: bytes) -> dict:
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

    # Generate spectrogram
    spec = generate_mel_spectrogram(audio_data, sr=sr)
    spec = spec[np.newaxis, ...]  # add channel dim: (1, n_mels, T)
    spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, 1, n_mels, T)

    model = get_model()
    probs = model.predict_proba(spec).squeeze().cpu().numpy().tolist()

    # Convert frame probabilities to regions
    frame_duration = 512 / sr  # hop_length / sr
    regions = []
    in_region = False
    start = 0.0
    conf_sum = 0.0
    count = 0

    for i, p in enumerate(probs):
        if p >= 0.5 and not in_region:
            in_region = True
            start = i * frame_duration
            conf_sum = p
            count = 1
        elif p >= 0.5 and in_region:
            conf_sum += p
            count += 1
        elif p < 0.5 and in_region:
            end = i * frame_duration
            conf = conf_sum / count if count > 0 else 0.0
            if (end - start) >= 0.1:
                regions.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": round(conf, 4),
                })
            in_region = False

    if in_region:
        end = len(probs) * frame_duration
        conf = conf_sum / count if count > 0 else 0.0
        if (end - start) >= 0.1:
            regions.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "confidence": round(conf, 4),
            })

    return {"regions": regions}
