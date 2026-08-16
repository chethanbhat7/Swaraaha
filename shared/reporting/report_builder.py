"""Shared Typst-based clinical report builder used by both the desktop and web apps."""

from datetime import date
from typing import Optional

import typst

REPORT_TITLE = "Swaraaha Stutter Analysis Report"

DISPLAY_NAMES = {
    "prolongation": "Prolongation",
    "block": "Block",
    "soundrep": "Sound Repetition",
    "wordrep": "Word Repetition",
    "interjection": "Interjection",
}


def _escape(text: object) -> str:
    """Escape Typst markup special characters in plain text values."""
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("#", "\\#")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("$", "\\$")
        .replace("\n", " ")
    )


def _display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def severity_for(data: dict) -> Optional[str]:
    """Stutter-index severity derived from the ``combined`` output.

    Returns None when ``combined`` is missing, errored, or has no duration, in
    which case callers render "N/A".
    """
    combined = data.get("combined")
    if not isinstance(combined, dict) or "error" in combined:
        return None
    regions = combined.get("regions") or []
    audio_duration = combined.get("audio_duration") or 0.0
    if audio_duration <= 0:
        return None
    coverage = sum(max(0.0, r.get("end", 0.0) - r.get("start", 0.0)) for r in regions)
    index = coverage / audio_duration * 100
    if index >= 15:
        return "Severe"
    if index >= 5:
        return "Moderate"
    if index >= 2:
        return "Mild"
    return "Fluent"


def _class_rows(classification: dict) -> str:
    rows = []
    for name, result in (classification or {}).items():
        if not isinstance(result, dict):
            continue
        confidence = result.get("confidence")
        if not isinstance(confidence, (int, float)):
            continue
        label = bool(result.get("label"))
        rows.append(
            f"  [{_escape(_display_name(name))}], "
            f"[{_escape('Detected' if label else 'Not Detected')}], "
            f"[{_escape(f'{confidence * 100:.1f}%')}]"
        )
    if not rows:
        return "  [No stuttering classes detected], [Not Detected], [0.0%]"
    return ",\n".join(rows)


def _region_rows(regions: list) -> str:
    rows = []
    for i, region in enumerate(regions, start=1):
        start = float(region.get("start", 0.0))
        end = float(region.get("end", 0.0))
        confidence = float(region.get("confidence", 0.0))
        ptype = region.get("primary_type")
        primary = _display_name(ptype) if ptype else "—"
        rows.append(
            f"  [{i}], [{_escape(f'{start:.2f}')}], [{_escape(f'{end:.2f}')}], "
            f"[{_escape(f'{max(0.0, end - start):.2f}')}], "
            f"[{_escape(f'{confidence * 100:.0f}%')}], [{_escape(primary)}]"
        )
    return ",\n".join(rows)


def build_report_source(data: dict) -> str:
    patient = data.get("patient") or {}
    audio = data.get("audio") or {}
    classification = data.get("classification") or {}
    transcription = data.get("transcription") or {}
    raw_combined = data.get("combined")
    combined = (
        raw_combined
        if isinstance(raw_combined, dict) and "error" not in raw_combined
        else None
    )

    date_text = _escape(data.get("date") or date.today().isoformat())

    transcript = (transcription.get("text") or "").strip()
    if not transcript:
        transcript = "No transcription available."

    severity = severity_for(data)
    severity_text = severity if severity else "N/A"

    total = combined.get("total_stutters") if combined else None
    total_text = _escape(str(total)) if total is not None else "—"

    regions = combined.get("regions") if combined else None
    if regions is None:
        fallback = data.get("localization")
        if isinstance(fallback, dict) and fallback.get("regions"):
            regions = fallback["regions"]
    if regions:
        region_block = rf"""#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr, 1.4fr),
  stroke: 0.5pt + rgb("#d0d0d0"),
  align: left,
  [*\#*], [*Start (s)*], [*End (s)*], [*Duration (s)*], [*Confidence*], [*Primary Type*],
{_region_rows(regions)}
)"""
    else:
        region_block = "No dysfluency events localized."

    return rf"""#set page(paper: "a4", margin: 2.5cm)
#set text(size: 11pt)

#align(center)[
  #text(size: 18pt, weight: "bold")[{REPORT_TITLE}]
]

#align(center)[
  #text(size: 10pt, fill: rgb("#555555"))[{date_text}]
]

#v(2em)

#text(size: 13pt, weight: "bold")[Patient Details]
#table(
  columns: 2,
  stroke: none,
  align: (left, left),
  [*Name*], [{_escape(patient.get('name') or 'N/A')}],
  [*Phone*], [{_escape(patient.get('phone') or 'N/A')}],
)

#v(1.5em)

#text(size: 13pt, weight: "bold")[Audio Details]
#table(
  columns: 2,
  stroke: none,
  align: (left, left),
  [*File Name*], [{_escape(audio.get('filename') or 'N/A')}],
  [*File Size*], [{_escape(audio.get('size') or 'N/A')}],
  [*Duration*], [{_escape(audio.get('duration') or 'N/A')}],
)

#v(1.5em)

#text(size: 13pt, weight: "bold")[Classification Results]
#table(
  columns: (1fr, 1.4fr, 1fr),
  stroke: 0.5pt + rgb("#d0d0d0"),
  align: left,
  [*Dysfluency Category*], [*Clinical Present Label*], [*Model Confidence Score*],
{_class_rows(classification)}
)

#v(1.5em)

#text(size: 13pt, weight: "bold")[Localized Dysfluency Events]
{region_block}

#v(1.5em)

#text(size: 13pt, weight: "bold")[Summary]
Total stuttering events: {total_text}
Overall severity: {severity_text}

#v(1.5em)

#text(size: 13pt, weight: "bold")[Transcript]
{_escape(transcript)}

#v(2em)

#align(center)[
  #text(size: 9pt, fill: rgb("#888888"))[
    Generated by Swaraaha. This report summarizes automated dysfluency detection
    output and is not a medical diagnosis; consult a qualified speech-language
    professional for a formal evaluation.
  ]
]
"""


def generate_report_pdf(data: dict) -> bytes:
    """Compile the unified report to PDF bytes using the typst CLI."""
    import tempfile
    from pathlib import Path

    source = build_report_source(data)
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "report.typ"
        source_path.write_text(source, encoding="utf-8")
        return typst.compile(source_path)
