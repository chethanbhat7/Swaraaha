"""Service layer for Wav2Vec2-based localization pipeline."""

import io
from typing import Optional

import numpy as np

from model.localization.ctc_alignment import CTCTimeAligner, SimpleForcedAligner
from model.localization.language_adapter import LanguageAdapterRegistry
from model.localization.wav2vec2_localizer import Wav2Vec2Localizer

_model: Optional[Wav2Vec2Localizer] = None
_aligner = None
_registry = None


def get_model() -> Wav2Vec2Localizer:
    global _model
    if _model is None:
        _model = Wav2Vec2Localizer()
    return _model


def get_aligner():
    global _aligner
    if _aligner is None:
        try:
            _aligner = CTCTimeAligner()
        except Exception:
            _aligner = SimpleForcedAligner()
    return _aligner


def get_registry():
    global _registry
    if _registry is None:
        _registry = LanguageAdapterRegistry()
    return _registry


def localize_audio_bytes_w2v2(
    audio_bytes: bytes,
    text: str = "",
    language_code: str = "en",
) -> dict:
    """
    Full localization pipeline with Wav2Vec2.

    Args:
        audio_bytes: Raw audio bytes.
        text: Optional transcript for word-level alignment.
        language_code: Language code for syllabification (en, kn, hi).

    Returns:
        {
            "regions": [{"start", "end", "confidence"}],
            "words": [{"word", "start", "end", "confidence"}],
            "syllables": [{"syllable", "start", "end", "word", "index", "total"}],
        }
    """
    import soundfile as sf

    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Normalize to float32 [-1, 1]
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    if np.abs(audio_data).max() > 1.0:
        audio_data = audio_data / (np.abs(audio_data).max() + 1e-8)

    # Truncate or pad to 10s
    if len(audio_data) > 160000:
        audio_data = audio_data[:160000]
    elif len(audio_data) < 160000:
        audio_data = np.pad(audio_data, (0, 160000 - len(audio_data)))

    # Frame-level localization
    model = get_model()
    regions = model.predict(audio_data, sr=sr, threshold=0.5)

    region_list = [
        {"start": round(s, 3), "end": round(e, 3), "confidence": round(c, 4)}
        for s, e, c in regions
    ]

    # Word-level alignment (if text provided)
    word_list = []
    syllable_list = []

    if text:
        try:
            aligner = get_aligner()
            word_timestamps = aligner.align(audio_data, text, sr=sr)

            word_list = [
                {"word": wt.word, "start": wt.start_sec, "end": wt.end_sec,
                 "confidence": round(wt.confidence, 4)}
                for wt in word_timestamps
            ]

            # Syllable-level adaptation
            registry = get_registry()
            adapter = registry.get(language_code)
            syllable_timestamps = adapter.adapt(word_timestamps, text)
            syllable_list = [
                {"syllable": s.syllable, "start": s.start_sec, "end": s.end_sec,
                 "word": s.word, "index": s.syllable_index, "total": s.total_syllables}
                for s in syllable_timestamps
            ]
        except Exception as e:
            print(f"Alignment/syllabification failed: {e}")

    return {
        "regions": region_list,
        "words": word_list,
        "syllables": syllable_list,
    }
