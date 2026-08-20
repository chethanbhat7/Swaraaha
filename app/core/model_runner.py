"""Model inference wrapper — classification, localization and transcription.

Uses model.init() + model.analyze() for all model operations.
Classification errors propagate so the UI surfaces them instead of
fabricating results.
"""

import logging

import numpy as np

from model import analyze as model_analyze

logger = logging.getLogger(__name__)


def _audio_to_bytes(audio: np.ndarray) -> bytes:
    """Convert a numpy audio array to WAV bytes for the model API."""
    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV")
    return buf.getvalue()


class ModelRunner:
    def __init__(self, models_dir: str = ""):
        self.models_dir = models_dir
        from app.core.transcription import AudioTranscriber
        self.transcriber = AudioTranscriber()

    def analyze(self, audio: np.ndarray, language: str = "english") -> dict:
        """Run classification + localization + transcription on audio. Returns structured results."""
        results = model_analyze(_audio_to_bytes(audio), language=language)

        classifications = {}
        cls = results.get("classification", {})
        if isinstance(cls, dict) and "error" not in cls:
            for name, result in cls.items():
                if name != "summary" and isinstance(result, dict):
                    classifications[name] = (bool(result.get("label", 0)), float(result.get("confidence", 0.0)))

        localizations = []
        loc = results.get("localization", {})
        if isinstance(loc, dict) and "error" not in loc:
            localizations = [
                (r["start"], r["end"], r["confidence"])
                for r in loc.get("regions", [])
            ]

        return {
            "classifications": classifications,
            "localizations": localizations,
            "transcription": results.get("transcription", {}),
            "combined": results.get("combined", {}),
        }
