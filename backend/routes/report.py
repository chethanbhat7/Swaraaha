"""Report API route — generates a shared Typst PDF clinical report."""

import re

from fastapi import APIRouter, HTTPException, Response

from shared.reporting.report_builder import generate_report_pdf

router = APIRouter()

REQUIRED_FIELDS = ("patient", "audio", "date", "classification")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-")


@router.post("/report")
async def generate_report(payload: dict):
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing field: {missing[0]}")
    try:
        pdf_bytes = generate_report_pdf(payload)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to generate PDF report") from exc

    filename = f"swaraaha-report-{_slugify(str(payload.get('date') or ''))}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
