#!/usr/bin/env python3
"""Generate the reading-document PDFs served by the frontend PDF viewer.

Renders each entry in frontend/src/data/readingDocuments.json into a PDF at
frontend/public/documents/<id>.pdf using the system typst compiler.

Usage:
    .venv/bin/python scripts/generate_docs.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "frontend" / "src" / "data" / "readingDocuments.json"
OUT_DIR = ROOT / "frontend" / "public" / "documents"
TYPST = shutil.which("typst") or "/home/mckb/.local/bin/typst"


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("#", "\\#")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("$", "\\$")
    )


def build_typ_source(doc: dict) -> str:
    sections = []
    for section in doc.get("sections", []):
        heading = _escape(str(section.get("heading", "")))
        content = section.get("content", "")
        if isinstance(content, list):
            content = "\n\n".join(str(line) for line in content)
        sections.append(
            f'#text(size: 13pt, weight: "bold")[{_escape(heading)}]\n\n{_escape(str(content))}'
        )
    return f"""#set page(paper: "a4", margin: 2cm)
#set text(size: 11pt, lang: "en")

#align(center)[
  #text(size: 16pt, weight: "bold")[{_escape(str(doc.get('title', '')))}]
]

#align(center)[
  #text(size: 10pt, fill: rgb("#555555"))[{_escape(str(doc.get('subtitle', '')))}]
]

#v(1.5em)

{chr(10).join(sections)}
"""


def main() -> int:
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    for doc in docs:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        out_path = OUT_DIR / f"{doc_id}.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            typ_path = Path(tmp) / "doc.typ"
            typ_path.write_text(build_typ_source(doc), encoding="utf-8")
            subprocess.run([TYPST, "compile", str(typ_path), str(out_path)], check=True)
            print(f"Generated {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
