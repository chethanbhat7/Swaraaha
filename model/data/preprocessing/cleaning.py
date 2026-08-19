"""Audio cleaning and normalization: DC removal, peak/RMS normalization, trimming."""

import numpy as np


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from audio signal.

    Subtracts the mean amplitude so the signal is centered around zero.
    This prevents bias in downstream feature extraction.

    Args:
        audio: 1-D float32 audio array.

    Returns:
        DC-corrected audio array.
    """
    return (audio - np.mean(audio)).astype(np.float32)


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to a target peak amplitude.

    Scales the signal so its absolute maximum equals target_peak.
    Prevents clipping while maximizing dynamic range.

    Args:
        audio: 1-D float32 audio array.
        target_peak: Target peak amplitude (0-1). Default 0.95.

    Returns:
        Peak-normalized audio array.
    """
    if audio.size == 0:
        return audio
    peak = np.abs(audio).max()
    if peak == 0:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    Normalize audio to a target RMS level.

    Scales the signal so its RMS equals target_rms.
    More perceptually uniform than peak normalization.

    Args:
        audio: 1-D float32 audio array.
        target_rms: Target RMS amplitude. Default 0.1.

    Returns:
        RMS-normalized audio array.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    return (audio * (target_rms / rms)).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int = 16000,
    top_db: int = 25,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Uses librosa's trim which detects non-silent intervals based on
    a dB threshold relative to the peak.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate (used for frame length).
        top_db: Threshold in dB below peak to consider as silence. Default 25.

    Returns:
        Trimmed audio array.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def trim_silence_center(
    audio: np.ndarray,
    sr: int = 16000,
    top_db: int = 25,
    pad_ms: float = 50.0,
) -> np.ndarray:
    """
    Trim silence and add a small padding around the speech.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        top_db: Silence threshold in dB.
        pad_ms: Padding in milliseconds to add around speech.

    Returns:
        Trimmed and padded audio array.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    pad_samples = int(pad_ms * sr / 1000)
    padded = np.pad(trimmed, (pad_samples, pad_samples), mode="constant")
    return padded.astype(np.float32)


def clean_audio(
    audio: np.ndarray,
    sr: int = 16000,
    remove_dc: bool = True,
    normalize: bool = True,
    trim: bool = True,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    Full audio cleaning pipeline: DC removal → normalization → silence trimming.

    This is the recommended preprocessing step before feeding audio to models.
    Apply this consistently across training and inference.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        remove_dc: Remove DC offset.
        normalize: Apply peak normalization.
        trim: Trim leading/trailing silence.
        target_peak: Target peak amplitude for normalization.

    Returns:
        Cleaned audio array.

    Example:
        >>> audio, sr = load_audio("raw_recording.wav")
        >>> audio = clean_audio(audio, sr=sr)
        >>> # audio is now DC-free, peak-normalized, silence-trimmed
    """
    if audio is None or len(audio) == 0:
        return audio
    if remove_dc:
        audio = remove_dc_offset(audio)
    if normalize:
        audio = normalize_peak(audio, target_peak=target_peak)
    if trim:
        audio = trim_silence(audio, sr=sr)
    return audio


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    p: float = 0.5,
) -> np.ndarray:
    """
    Apply random augmentation pipeline to audio with probability p.

    Convenience wrapper around AudioAugmentor for use in dataset __getitem__.

    Args:
        audio: 1-D float32 audio array, values in [-1.0, 1.0].
        sr: Sample rate in Hz.
        p: Probability of applying augmentation (0.0 = no augmentation).

    Returns:
        Augmented (or original) audio array, same length as input.

    Example:
        >>> # In dataset __getitem__:
        >>> if self.augment:
        >>>     audio = augment_audio(audio, sr=self.sr, p=0.5)
    """
    import random

    from model.data.augmentation import AudioAugmentor

    if random.random() > p:
        return audio

    augmentor = AudioAugmentor()
    return augmentor(audio, sr=sr)
