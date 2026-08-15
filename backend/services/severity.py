"""Diagnostic severity scoring for Swaraaha."""

from typing import Any, Dict, List

SEVERITY_LABELS = {
    "fluent": "Fluent",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}


def compute_severity(regions: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
    """Compute stutter index and severity from localized dysfluency regions.

    Stutter index = total dysfluency coverage / speech duration * 100.
    """
    coverage = sum(max(0.0, float(region["end"]) - float(region["start"])) for region in (regions or []))
    index_pct = (coverage / duration_sec * 100.0) if duration_sec and duration_sec > 0 else 0.0

    if index_pct >= 15:
        severity = "severe"
    elif index_pct >= 5:
        severity = "moderate"
    elif index_pct >= 2:
        severity = "mild"
    else:
        severity = "fluent"

    return {
        "index_pct": round(index_pct, 2),
        "severity": severity,
        "label": SEVERITY_LABELS[severity],
    }
