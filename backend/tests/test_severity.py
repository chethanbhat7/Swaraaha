from backend.services.severity import compute_severity


def test_fluent_below_two_percent():
    assert compute_severity([], 10.0)["severity"] == "fluent"
    assert compute_severity([{"start": 0, "end": 0.1}], 10.0)["severity"] == "fluent"


def test_mild():
    result = compute_severity([{"start": 0, "end": 0.3}], 10.0)
    assert result["severity"] == "mild"
    assert result["label"] == "Mild"


def test_moderate():
    assert compute_severity([{"start": 0, "end": 1.0}], 10.0)["severity"] == "moderate"


def test_severe():
    result = compute_severity([{"start": 0, "end": 2.0}], 10.0)
    assert result["severity"] == "severe"
    assert result["index_pct"] == 20.0


def test_zero_duration_guard():
    assert compute_severity([{"start": 0, "end": 2.0}], 0.0)["index_pct"] == 0.0
