"""
document_processor.py
Extract and clean text from medical PDF reports.
Memory-safe: limits text length to prevent MemoryError on large PDFs.
"""

import re

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Max pages and text length to prevent MemoryError on huge PDFs
MAX_PAGES   = 30
MAX_CHARS   = 150_000


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Try PyMuPDF first; fall back to pdfplumber.
    Limits to MAX_PAGES pages and MAX_CHARS characters.
    """
    text = ""

    if PYMUPDF_AVAILABLE:
        text = _extract_with_pymupdf(pdf_path)

    if (not text or len(text.strip()) < 100) and PDFPLUMBER_AVAILABLE:
        text = _extract_with_pdfplumber(pdf_path)

    if not text:
        raise RuntimeError(
            "Could not extract text from PDF. "
            "Make sure pymupdf and pdfplumber are installed."
        )

    # Hard cap on text length — lab reports never need more than this
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    return text


def _extract_with_pymupdf(pdf_path: str) -> str:
    try:
        doc   = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            if i >= MAX_PAGES:
                break
            pages.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"[PyMuPDF Error] {e}")
        return ""


def _extract_with_pdfplumber(pdf_path: str) -> str:
    try:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= MAX_PAGES:
                    break
                t = page.extract_text()
                if t:
                    pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        print(f"[pdfplumber Error] {e}")
        return ""


def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw extracted text.
    """
    # Keep ASCII + Devanagari (Hindi) only
    text = re.sub(r'[^\x00-\x7F\u0900-\u097F]+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove repeated special chars like ----
    text = re.sub(r'[_\-]{3,}', ' ', text)
    text = text.replace('\n', ' ').strip()
    return text
