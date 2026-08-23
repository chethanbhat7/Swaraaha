"""Background transcription worker for the home page transcript."""

import numpy as np
from PySide6.QtCore import QThread, Signal

from model import transcribe as model_transcribe


class TranscriptionWorker(QThread):
    """Runs model.transcribe() off the UI thread and emits finished(dict)."""

    finished = Signal(dict)

    def __init__(self, audio: np.ndarray, language: str = "english"):
        super().__init__()
        self._audio = audio
        self._language = language

    def run(self):
        try:
            data = model_transcribe(self._audio, language=self._language)
            self.finished.emit(data)
        except Exception as e:
            self.finished.emit({"error": str(e)})
