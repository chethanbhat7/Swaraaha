"""Model inference wrapper — stub for Phase 1. Will integrate with model/ in Phase 3."""

import numpy as np


from app.core.transcription import AudioTranscriber


class ModelRunner:
    def __init__(self, models_dir: str = ""):
        self.models_dir = models_dir
        self._loaded = False
        self.transcriber = AudioTranscriber()

    def transcribe(self, audio: np.ndarray, localizations=None) -> dict:
        """Run transcription pipeline on audio."""
        return self.transcriber.transcribe(audio, localizations=localizations)

    def analyze(self, audio: np.ndarray) -> dict:
        """Run classification + localization + transcription on audio. Returns structured results."""
        localizations = [
            (0.5, 1.2, 0.87),
            (3.4, 4.1, 0.72),
        ]
        classifications = {
            "prolongation": (False, 0.12),
            "block": (True, 0.87),
            "soundrep": (False, 0.08),
            "wordrep": (False, 0.05),
            "interjection": (True, 0.72),
        }
        transcription = self.transcribe(audio, localizations=localizations)

        return {
            "classifications": classifications,
            "localizations": localizations,
            "transcription": transcription,
        }
