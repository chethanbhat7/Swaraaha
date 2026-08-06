"""Background transcription worker for the home page transcript."""

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.core.transcription import AudioTranscriber


class TranscriptionWorker(QThread):
    """Runs AudioTranscriber.transcribe off the UI thread and emits finished(dict)."""

    finished = Signal(dict)

    def __init__(self, transcriber: AudioTranscriber, audio: np.ndarray, language: str = "english"):
        super().__init__()
        self._transcriber = transcriber
        self._audio = audio
        self._language = language

    def run(self):
        try:
            data = self._transcriber.transcribe(self._audio, language=self._language)
            self.finished.emit(data)
        except Exception as e:
            self.finished.emit({"error": str(e)})
