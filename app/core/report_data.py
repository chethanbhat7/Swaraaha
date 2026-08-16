"""Normalize ModelRunner results into the shared report data contract."""

from datetime import date


def build_report_data(
    results: dict,
    *,
    patient_name: str = "",
    filename: str = "",
    duration_sec: float = 0.0,
) -> dict:
    """Build the shared-report data dict from desktop analysis results.

    ``classifications`` are ``{name: (stutter_present, confidence)}`` tuples;
    ``localizations`` are ``(start, end, confidence)`` tuples.
    """
    classifications = results.get("classifications", {})
    localizations = results.get("localizations", [])
    return {
        "patient": {"name": patient_name, "phone": ""},
        "audio": {"filename": filename, "size": "", "duration": f"{duration_sec:.2f} s"},
        "date": date.today().isoformat(),
        "classification": {
            name: {"label": int(detected), "confidence": float(confidence)}
            for name, (detected, confidence) in classifications.items()
        },
        "combined": results.get("combined"),
        "transcription": results.get("transcription") or {},
        "localization": {
            "regions": [
                {"start": s, "end": e, "confidence": c}
                for s, e, c in localizations
            ]
        },
    }
