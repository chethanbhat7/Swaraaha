"""Typst-based clinical PDF report generation."""

import tempfile
from pathlib import Path

import typst

REPORT_TITLE = "Swaraaha Stutter Analysis Report"


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


def build_typ_source(data: dict) -> str:
    patient = data.get("patient") or {}
    audio = data.get("audio") or {}
    classification = data.get("classification") or {}
    date_text = _escape(data.get("date") or "")

    rows = []
    for name, result in classification.items():
        if not isinstance(result, dict):
            continue
        confidence = result.get("confidence")
        if not isinstance(confidence, (int, float)):
            continue
        label = bool(result.get("label"))
        rows.append(
            f"  [{_escape(name)}], [{_escape('Detected' if label else 'Not Detected')}], "
            f"[{_escape(f'{confidence * 100:.1f}%')}]"
        )

    class_rows = ",\n".join(rows) if rows else "  [No stuttering classes detected], [Not Detected], [0.0%]"

    return f"""#set page(paper: "a4", margin: 2.5cm)
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

#text(size: 13pt, weight: "bold")[Stuttering Classes]
#table(
  columns: (1fr, 1.4fr, 1fr),
  stroke: 0.5pt + rgb("#d0d0d0"),
  align: left,
  [*Dysfluency Category*], [*Clinical Present Label*], [*Model Confidence Score*],
{class_rows}
)
"""


def generate_report_pdf(data: dict) -> bytes:
    source = build_typ_source(data)
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "report.typ"
        source_path.write_text(source, encoding="utf-8")
        return typst.compile(source_path)
