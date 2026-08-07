import numpy as np

from app.ui.transcription_worker import TranscriptionWorker


class _FakeTranscriber:
    def transcribe(self, audio, language="english"):
        return {"text": "hello", "words": [], "language": language}


def test_transcription_worker_emits_data(qapp):
    results = []
    worker = TranscriptionWorker(_FakeTranscriber(), np.zeros(1600, dtype=np.float32), "kannada")
    worker.finished.connect(results.append)
    worker.run()
    assert results[0]["text"] == "hello"
    assert results[0]["language"] == "kannada"


class _RaisingTranscriber:
    def transcribe(self, audio, language="english"):
        raise RuntimeError("boom")


def test_transcription_worker_emits_error(qapp):
    results = []
    worker = TranscriptionWorker(_RaisingTranscriber(), np.zeros(1600, dtype=np.float32))
    worker.finished.connect(results.append)
    worker.run()
    assert results[0]["error"] == "boom"
