import numpy as np
import pytest

from model import ModelRegistry
from model.registry import Classifier


def test_run_all_composes(monkeypatch):
    reg = ModelRegistry()

    monkeypatch.setattr(
        reg.classifier, "analyze",
        lambda audio, threshold=None: {"prolongation": {"label": 0}, "summary": {"detected": []}},
    )
    monkeypatch.setattr(
        reg.localizer, "analyze",
        lambda audio, text=None, language="en", threshold=0.3, max_length_seconds=10.0: {
            "regions": [{"start": 0.0, "end": 0.5, "confidence": 0.9}]
        },
    )
    monkeypatch.setattr(
        reg.transcriber, "transcribe",
        lambda audio, language="english", localizations=None, passage_text=None, sample_rate=16000: {
            "text": "hello world", "words": [], "duration_sec": 1.0
        },
    )

    result = reg.run_all(np.zeros(16000, dtype=np.float32), text="hello world")
    assert "classification" in result
    assert result["localization"]["regions"]
    assert result["transcription"]["text"] == "hello world"


def test_run_all_language_maps_to_iso(monkeypatch):
    reg = ModelRegistry()
    seen = {}

    def fake_analyze(audio, text=None, language="en", threshold=0.3, max_length_seconds=10.0):
        seen["language"] = language
        return {"regions": []}

    monkeypatch.setattr(reg.localizer, "analyze", fake_analyze)
    monkeypatch.setattr(
        reg.classifier, "analyze",
        lambda audio, threshold=None: {"summary": {"detected": []}},
    )
    monkeypatch.setattr(
        reg.transcriber, "transcribe",
        lambda audio, language="english", localizations=None, passage_text=None, sample_rate=16000: {
            "text": "", "words": [], "duration_sec": 0.0
        },
    )

    reg.run_all(np.zeros(16000, dtype=np.float32), language="english", text="hello")
    assert seen["language"] == "en"


def test_run_all_catches_missing_models(monkeypatch):
    reg = ModelRegistry()

    def _raise(*a, **k):
        raise FileNotFoundError("no")

    monkeypatch.setattr(reg.classifier, "analyze", _raise)
    monkeypatch.setattr(reg.localizer, "analyze", _raise)
    monkeypatch.setattr(
        reg.transcriber, "transcribe",
        lambda audio, **kwargs: {"text": "", "words": [], "duration_sec": 0.0},
    )
    result = reg.run_all(np.zeros(16000, dtype=np.float32))
    assert result["classification"]["error"]
    assert result["localization"]["error"]


def test_run_all_catches_arbitrary_errors(monkeypatch):
    """run_all must degrade classification/localization sub-results to
    {"error": ...} for ANY exception, not just FileNotFoundError (e.g. a
    ValueError from empty audio), mirroring the transcription handler."""
    reg = ModelRegistry()

    def _raise(*a, **k):
        raise ValueError("empty audio")

    monkeypatch.setattr(reg.classifier, "analyze", _raise)
    monkeypatch.setattr(reg.localizer, "analyze", _raise)
    monkeypatch.setattr(
        reg.transcriber, "transcribe",
        lambda audio, **kwargs: {"text": "", "words": [], "duration_sec": 0.0},
    )
    result = reg.run_all(np.zeros(16000, dtype=np.float32))
    assert result["classification"]["error"] == "empty audio"
    assert result["localization"]["error"] == "empty audio"
    assert result["transcription"]["text"] == ""


def test_classifier_all_mode_skips_unknown_registry_entries(monkeypatch, tmp_path):
    canonical = {"prolongation", "block", "soundrep", "wordrep", "interjection"}

    classification = {}
    for name in canonical:
        path = tmp_path / f"{name}.pt"
        path.write_bytes(b"")
        classification[name] = str(path)
    clut_path = tmp_path / "cluttering.pt"
    clut_path.write_bytes(b"")
    classification["cluttering"] = str(clut_path)

    registry = {"classification": classification}
    monkeypatch.setattr("model.registry._load_registry", lambda: registry)

    class _Stub:
        def predict(self, audio_tensor, threshold=0.5):
            return (1, 0.9)

    def fake_load_classifier(class_name, path):
        if class_name not in canonical:
            raise ValueError(f"Unknown dysfluency class: {class_name}")
        return _Stub()

    monkeypatch.setattr("model.registry._load_classifier", fake_load_classifier)

    clf = Classifier()
    clf._load()

    assert set(clf._models) == canonical
    assert "cluttering" not in clf._models
    out = clf.predict_all(np.zeros(16000, dtype=np.float32))
    assert set(out) == canonical


class _ThresholdAwareStub:
    def __init__(self, prob):
        self.prob = prob

    def predict(self, audio_tensor, threshold=0.5):
        label = 1 if self.prob >= threshold else 0
        confidence = self.prob if label == 1 else 1.0 - self.prob
        return (label, confidence)


def test_classifier_analyze_empty_audio_returns_empty_result(monkeypatch):
    """Classifier.analyze must not crash on empty audio; returns a well-formed
    'not present' result instead, without loading any model (mirrors the
    Transcriber's empty-audio guard)."""
    def _raise(*a, **k):
        raise AssertionError("must not load models for empty audio")

    monkeypatch.setattr(Classifier, "_load", _raise)
    clf = Classifier(class_name="prolongation")
    out = clf.analyze(np.array([], dtype=np.float32))
    assert out["label"] == 0
    assert out["confidence"] == 0.0
    assert out["prob_present"] == 0.0
    assert out["prob_not_present"] == 1.0


def test_classifier_analyze_none_audio_returns_empty_result(monkeypatch):
    monkeypatch.setattr(Classifier, "_load", lambda self: None)
    clf = Classifier(class_name="block")
    out = clf.analyze(None)
    assert out["label"] == 0
    assert out["confidence"] == 0.0


def test_classifier_analyze_raw_empty_includes_logits(monkeypatch):
    monkeypatch.setattr(Classifier, "_load", lambda self: None)
    clf = Classifier(class_name="block")
    out = clf.analyze_raw(np.array([], dtype=np.float32))
    assert out["logits"] == {"not_present": 0.0, "present": 0.0}


def test_classifier_analyze_empty_audio_all_mode(monkeypatch):
    registry = {"classification": {"prolongation": "x.pt", "block": "y.pt"}}
    monkeypatch.setattr("model.registry._load_registry", lambda: registry)

    clf = Classifier()
    out = clf.analyze(np.array([], dtype=np.float32))
    assert set(out) == {"prolongation", "block", "summary"}
    assert out["summary"] == {"detected": [], "primary": "prolongation"}
    for name in ("prolongation", "block"):
        assert out[name]["label"] == 0
        assert out[name]["prob_present"] == 0.0
        assert out[name]["prob_not_present"] == 1.0


def test_classifier_predict_honors_threshold(monkeypatch, tmp_path):
    path = tmp_path / "prolongation.pt"
    path.write_bytes(b"")
    registry = {"classification": {"prolongation": str(path)}}
    monkeypatch.setattr("model.registry._load_registry", lambda: registry)
    monkeypatch.setattr(
        "model.registry._load_classifier",
        lambda name, p: _ThresholdAwareStub(0.7),
    )

    clf = Classifier(class_name="prolongation")
    audio = np.zeros(16000, dtype=np.float32)
    assert clf.predict(audio, threshold=0.8)[0] == 0
    assert clf.predict(audio, threshold=0.5)[0] == 1


def test_classifier_predict_all_honors_thresholds(monkeypatch, tmp_path):
    canonical = {"prolongation", "block", "soundrep", "wordrep", "interjection"}
    classification = {}
    for name in canonical:
        path = tmp_path / f"{name}.pt"
        path.write_bytes(b"")
        classification[name] = str(path)
    registry = {
        "classification": classification,
        "thresholds": {"prolongation": 0.8},
    }
    monkeypatch.setattr("model.registry._load_registry", lambda: registry)
    monkeypatch.setattr(
        "model.registry._load_classifier",
        lambda name, p: _ThresholdAwareStub(0.7),
    )

    clf = Classifier()
    audio = np.zeros(16000, dtype=np.float32)
    out = clf.predict_all(audio)
    assert out["prolongation"][0] == 0
    assert out["block"][0] == 1

    out2 = clf.predict_all(audio, threshold=0.6)
    assert out2["prolongation"][0] == 1
    assert out2["block"][0] == 1


def test_multitask_classifier_analyze_returns_per_class_output(tmp_path, monkeypatch):
    """MultiTaskClassifier.analyze returns per-class dicts + summary."""
    import torch

    from model.registry import MultiTaskClassifier

    class _FakeHeads(torch.nn.Module):
        def forward(self, pooled):
            out = {}
            for i, name in enumerate(
                ["prolongation", "block", "soundrep", "wordrep", "interjection"]
            ):
                out[name] = torch.tensor([[0.0, -3.0]]) if i != 1 else torch.tensor([[0.0, 3.0]])
            return out

    class _FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = _FakeHeads()
            self.config = type("C", (), {"hidden_size": 8})()

        def forward(self, input_values):
            return self.heads(input_values)

    class _FakeModel:
        def __init__(self):
            self.model = _FakeBackbone()

        def forward(self, input_values):
            return self.model(input_values)

    import model.classification.multitask as mt

    monkeypatch.setattr(mt, "_wav2vec2_model_class", lambda: type("F", (), {
        "from_pretrained": staticmethod(lambda model_name: _FakeBackbone())
    }))

    # Point registry at a dummy checkpoint and stub the loader.
    import json

    reg_path = str(tmp_path / "registry.json")
    with open(reg_path, "w") as f:
        json.dump({"classification_multitask": {"path": "weights/mt.pt",
                                                 "model_name": "fake"}}, f)
    monkeypatch.setattr("model.registry._REGISTRY_PATH", reg_path)
    monkeypatch.setattr("model.registry._resolve_path", lambda p: str(tmp_path / "mt.pt"))
    (tmp_path / "mt.pt").write_bytes(b"dummy")
    monkeypatch.setattr("model.registry._load_multitask_classifier",
                        lambda path: _FakeModel())

    clf = MultiTaskClassifier()
    out = clf.analyze(np.zeros(1600, dtype=np.float32))

    assert out["block"]["label"] == 1
    assert out["prolongation"]["label"] == 0
    assert out["summary"]["detected"] == ["block"]
    assert out["summary"]["primary"] == "block"


def test_multitask_classifier_analyze_empty_audio_returns_empty_result(tmp_path, monkeypatch):
    """Empty audio must not load models or crash (mirrors M19 guards)."""
    from model.registry import MultiTaskClassifier

    called = {"loads": 0}

    def fake_load():
        called["loads"] += 1
        raise AssertionError("model should not be loaded for empty audio")

    clf = MultiTaskClassifier()
    monkeypatch.setattr(clf, "_load", fake_load)

    out = clf.analyze(None)
    assert called["loads"] == 0
    assert out["summary"] == {"detected": [], "primary": "prolongation"}
    assert all(r["label"] == 0 for r in out.values() if r != out["summary"])
