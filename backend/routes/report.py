"""Report API route — generates a Typst PDF clinical report."""

import re

from fastapi import APIRouter, HTTPException, Response

from backend.services.report_generator import generate_report_pdf

router = APIRouter()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-")


@router.post("/report")
async def generate_report(payload: dict):
    try:
        pdf_bytes = generate_report_pdf(payload)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to generate PDF report") from exc

    filename = f"swaraaha-report-{_slugify(str(payload.get('date') or ''))}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
