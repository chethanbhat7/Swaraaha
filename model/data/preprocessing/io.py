"""Audio I/O: loading, conversion, and input handling."""

import os
import subprocess
from typing import Tuple

import numpy as np


def convert_to_wav(audio_bytes: bytes) -> bytes:
    """Convert arbitrary audio bytes to standard 16kHz mono WAV format using ffmpeg."""
    try:
        process = subprocess.Popen(
            ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-ar', '16000', '-ac', '1', 'pipe:1'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=audio_bytes)
        if process.returncode != 0:
            print(f"ffmpeg error: {stderr.decode('utf-8', errors='ignore')}")
            return audio_bytes
        return stdout
    except Exception as e:
        print(f"Failed to convert audio via ffmpeg: {e}")
        return audio_bytes


def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        path: Path to audio file (.wav, .flac, .mp3, etc.).
        sr: Target sample rate in Hz. Default 16000 (matches Wav2Vec 2.0).

    Returns:
        Tuple of (audio_array, sample_rate).
        audio_array is 1-D numpy float32 array with values in [-1.0, 1.0].

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio file cannot be decoded.

    Example:
        >>> audio, sr = load_audio("recording.wav")
        >>> print(f"Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}")
    """
    import librosa

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32), sr


def load_audio_from_array(audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Accept a raw audio numpy array (e.g. from desktop app recording) and
    ensure it matches the expected format.

    Args:
        audio: 1-D numpy array of audio samples.
        sr: Sample rate of the input audio.

    Returns:
        Tuple of (resampled_audio, target_sr).
    """
    import librosa

    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D audio array, got shape {audio.shape}")

    if len(audio) == 0:
        return audio, sr

    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    return audio, sr


def load_audio_input(audio, sr: int = 16000) -> np.ndarray:
    """Load raw audio from a file path, bytes, or numpy array.

    Args:
        audio: File path (str/PathLike), raw bytes, or 1-D numpy array.
        sr: Target sample rate in Hz (16000 for Wav2Vec2-based models).

    Returns:
        1-D float32 mono numpy array, values in [-1.0, 1.0].

    Raises:
        FileNotFoundError: If the path does not exist or cannot be decoded.
        ValueError: If an array is not 1-D.
        TypeError: If the input type is not supported.
    """
    import io
    import os

    import soundfile as sf

    if isinstance(audio, (str, os.PathLike)):
        if not os.path.isfile(audio):
            raise FileNotFoundError(f"Audio file not found: {audio}")
        audio_array, _ = load_audio(audio, sr=sr)
    elif isinstance(audio, bytes):
        wav_bytes = convert_to_wav(audio)
        audio_array, file_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if file_sr != sr:
            import librosa

            audio_array = librosa.resample(audio_array, orig_sr=file_sr, target_sr=sr)
    elif isinstance(audio, np.ndarray):
        audio_array, _ = load_audio_from_array(audio, sr=sr)
    else:
        raise TypeError(
            f"Unsupported audio type: {type(audio).__name__}. "
            "Expected str path, bytes, or numpy array."
        )
    return audio_array
