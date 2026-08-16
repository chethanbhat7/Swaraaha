"""Tests for the backend classification service using the multitask classifier."""

from backend.services import classifier as classifier_service


def test_classify_audio_bytes_falls_back_to_per_class_classifier(monkeypatch):
    captured = {}

    class _FakePerClass:
        def __init__(self):
            captured["instance"] = self

        def analyze(self, audio, threshold=None):
            captured["audio"] = audio
            return {
                "prolongation": {"label": 1, "confidence": 0.9},
                "block": {"label": 0, "confidence": 0.7},
                "summary": {
                    "detected": ["prolongation"],
                    "primary": "prolongation",
                },
            }

    class _FakeMultiTask:
        def analyze(self, audio, threshold=None):
            raise FileNotFoundError("Model file not found")

    monkeypatch.setattr(classifier_service, "MultiTaskClassifier", _FakeMultiTask)
    monkeypatch.setattr(classifier_service, "Classifier", _FakePerClass)
    monkeypatch.setattr(classifier_service, "_clf", None)

    result = classifier_service.classify_audio_bytes(b"fake-wav-bytes")

    assert isinstance(captured["instance"], _FakePerClass)
    assert captured["audio"] == b"fake-wav-bytes"
    assert result["prolongation"] == {"label": 1, "confidence": 0.9}
    assert result["summary"] == {"detected": ["prolongation"], "primary": "prolongation"}


def test_classify_audio_bytes_fallback_persists_across_calls(monkeypatch):
    captured = {"calls": 0}

    class _FakePerClass:
        def analyze(self, audio, threshold=None):
            captured["calls"] += 1
            return {"summary": {"detected": [], "primary": "prolongation"}}

    class _FakeMultiTask:
        def analyze(self, audio, threshold=None):
            raise FileNotFoundError("Model file not found")

    monkeypatch.setattr(classifier_service, "MultiTaskClassifier", _FakeMultiTask)
    monkeypatch.setattr(classifier_service, "Classifier", _FakePerClass)
    monkeypatch.setattr(classifier_service, "_clf", None)

    classifier_service.classify_audio_bytes(b"fake-wav-bytes")
    classifier_service.classify_audio_bytes(b"fake-wav-bytes")

    assert captured["calls"] == 2
    assert isinstance(classifier_service._clf, _FakePerClass)


def test_classify_audio_bytes_delegates_to_multitask_classifier(monkeypatch):
    captured = {}

    class _FakeMultiTask:
        def __init__(self):
            captured["instance"] = self

        def analyze(self, audio, threshold=None):
            captured["audio"] = audio
            return {
                "prolongation": {"label": 1, "confidence": 0.9},
                "block": {"label": 0, "confidence": 0.7},
                "summary": {
                    "detected": ["prolongation"],
                    "primary": "prolongation",
                },
            }

    monkeypatch.setattr(classifier_service, "MultiTaskClassifier", _FakeMultiTask)
    monkeypatch.setattr(classifier_service, "_clf", None)

    result = classifier_service.classify_audio_bytes(b"fake-wav-bytes")

    assert isinstance(captured["instance"], _FakeMultiTask)
    assert captured["audio"] == b"fake-wav-bytes"
    assert result["prolongation"] == {"label": 1, "confidence": 0.9}
    assert result["block"] == {"label": 0, "confidence": 0.7}
    assert result["summary"] == {"detected": ["prolongation"], "primary": "prolongation"}
