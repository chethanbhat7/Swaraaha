"""PDF text extraction using pdfplumber. Pure logic, no Qt imports."""

import os
import pdfplumber


class PdfHandler:
    @staticmethod
    def extract_text(path: str) -> str:
        """Extract plain text from a PDF file."""
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)

    @staticmethod
    def list_pdfs(directory: str) -> list[str]:
        """List all .pdf files in the given directory (non-recursive)."""
        if not os.path.isdir(directory):
            return []
        return sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith(".pdf")
            ]
        )
