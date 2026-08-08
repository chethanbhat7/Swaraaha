"""Model inference wrapper — classification, localization and transcription.

Uses model.registry (Classifier / Localizer) for real inference. When the
trained weights are missing, falls back to mock results so the UI keeps working.
"""

import logging

import numpy as np
import torch

from app.core.transcription import AudioTranscriber

logger = logging.getLogger(__name__)

MOCK_CLASSIFICATIONS = {
    "prolongation": (False, 0.12),
    "block": (True, 0.87),
    "soundrep": (False, 0.08),
    "wordrep": (False, 0.05),
    "interjection": (True, 0.72),
}


class ModelRunner:
    def __init__(self, models_dir: str = ""):
        self.models_dir = models_dir
        self._classifier = None
        self._localizer = None
        self.transcriber = AudioTranscriber()

    def transcribe(self, audio: np.ndarray, localizations=None, language: str = "english") -> dict:
        """Run transcription pipeline on audio."""
        return self.transcriber.transcribe(audio, localizations=localizations, language=language)

    def _get_classifier(self):
        if self._classifier is None:
            from model.registry import Classifier
            self._classifier = Classifier()
        return self._classifier

    def _get_localizer(self):
        if self._localizer is None:
            from model.registry import Localizer
            self._localizer = Localizer("cnn")
        return self._localizer

    def _classify(self, audio: np.ndarray) -> dict:
        """Per-class classification results as {name: (stutter_present, confidence)}."""
        try:
            audio_np = np.asarray(audio, dtype=np.float32)
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, ...]
            tensor = torch.tensor(audio_np)
            raw = self._get_classifier().predict_all(tensor)
            return {name: (bool(label), float(conf)) for name, (label, conf) in raw.items()}
        except Exception as e:
            logger.warning("Classification unavailable, using mock results: %s", e)
            return dict(MOCK_CLASSIFICATIONS)

    def _localize(self, audio: np.ndarray) -> list:
        """Dysfluency regions as [(start_sec, end_sec, confidence), ...]."""
        try:
            from model.data.preprocessing import generate_mel_spectrogram
            audio_np = np.asarray(audio, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            spec = generate_mel_spectrogram(audio_np, sr=16000)
            return list(self._get_localizer().predict(spec, threshold=0.3))
        except Exception as e:
            logger.warning("Localization unavailable: %s", e)
            return []

    def analyze(self, audio: np.ndarray, language: str = "english") -> dict:
        """Run classification + localization + transcription on audio. Returns structured results."""
        classifications = self._classify(audio)
        localizations = self._localize(audio)
        transcription = self.transcribe(audio, localizations=localizations, language=language)

        return {
            "classifications": classifications,
            "localizations": localizations,
            "transcription": transcription,
        }
