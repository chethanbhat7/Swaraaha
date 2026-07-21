"""Audio to mel-spectrogram conversion for the localization CNN."""

import numpy as np
import torch


def audio_to_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    tensor = torch.tensor(mel_db, dtype=torch.float32)
    return tensor.unsqueeze(0)


def load_audio(
    path: str,
    sr: int = 16000,
    duration: float = 10.0,
) -> np.ndarray:
    import librosa

    target_length = int(sr * duration)
    audio, _ = librosa.load(path, sr=sr, duration=duration)

    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    return audio
