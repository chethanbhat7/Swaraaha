import numpy as np

from app.ui.transcription_worker import TranscriptionWorker


def test_transcription_worker_emits_data(qapp, monkeypatch):
    monkeypatch.setattr(
        "app.ui.transcription_worker.model_transcribe",
        lambda audio, language="english", **kw: {"text": "hello", "words": [], "language": language},
    )
    results = []
    worker = TranscriptionWorker(np.zeros(1600, dtype=np.float32), "kannada")
    worker.finished.connect(results.append)
    worker.run()
    assert results[0]["text"] == "hello"
    assert results[0]["language"] == "kannada"


def test_transcription_worker_emits_error(qapp, monkeypatch):
    def _raise(audio, language="english", **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.ui.transcription_worker.model_transcribe", _raise)
    results = []
    worker = TranscriptionWorker(np.zeros(1600, dtype=np.float32))
    worker.finished.connect(results.append)
    worker.run()
    assert results[0]["error"] == "boom"
