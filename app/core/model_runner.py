"""Model inference wrapper — classification, localization and transcription.

Uses model.registry high-level pipelines for all model operations.
Classification errors propagate so the UI surfaces them instead of
fabricating results.
"""

import logging

import numpy as np

from app.core.transcription import AudioTranscriber
from model.registry import (
    classify_audio_bytes,
    combine_with_saliency,
    localize_audio_bytes,
)

logger = logging.getLogger(__name__)


def _audio_to_bytes(audio: np.ndarray) -> bytes:
    """Convert a numpy audio array to WAV bytes for the registry pipelines."""
    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV")
    return buf.getvalue()


class ModelRunner:
    def __init__(self, models_dir: str = ""):
        self.models_dir = models_dir
        self.transcriber = AudioTranscriber()

    def transcribe(self, audio: np.ndarray, localizations=None, language: str = "english") -> dict:
        """Run transcription pipeline on audio."""
        return self.transcriber.transcribe(audio, localizations=localizations, language=language)

    def _classify(self, audio: np.ndarray) -> dict:
        """Multi-task classification results as {name: (stutter_present, confidence)}."""
        raw = classify_audio_bytes(_audio_to_bytes(audio))
        return {
            name: (bool(result["label"]), float(result["confidence"]))
            for name, result in raw.items()
            if name != "summary"
        }

    def _localize(self, audio: np.ndarray) -> list:
        """Dysfluency regions as [(start_sec, end_sec, confidence), ...]."""
        try:
            result = localize_audio_bytes(_audio_to_bytes(audio))
            return [
                (r["start"], r["end"], r["confidence"])
                for r in result.get("regions", [])
            ]
        except Exception as e:
            logger.warning("Localization unavailable: %s", e)
            return []

    def _combine(self, audio: np.ndarray, localizations: list) -> dict:
        """Fuse localization tuples with classifier saliency into combined regions."""
        try:
            regions = [
                {"start": s, "end": e, "confidence": c}
                for s, e, c in localizations
            ]
            return combine_with_saliency(_audio_to_bytes(audio), regions)
        except Exception as exc:
            logger.warning("Combined fusion unavailable: %s", exc)
            return {"error": str(exc)}

    def analyze(self, audio: np.ndarray, language: str = "english") -> dict:
        """Run classification + localization + transcription on audio. Returns structured results."""
        classifications = self._classify(audio)
        localizations = self._localize(audio)
        transcription = self.transcribe(audio, localizations=localizations, language=language)
        combined = self._combine(audio, localizations)

        return {
            "classifications": classifications,
            "localizations": localizations,
            "transcription": transcription,
            "combined": combined,
        }
