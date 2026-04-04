"""
document_processor.py
Extract and clean text from medical PDF reports.
"""

import re
import fitz  # PyMuPDF
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Try PyMuPDF first; fall back to pdfplumber for scanned/complex PDFs.
    """
    text = _extract_with_pymupdf(pdf_path)
    if not text or len(text.strip()) < 100:
        text = _extract_with_pdfplumber(pdf_path)
    return text


def _extract_with_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def _extract_with_pdfplumber(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n".join(pages)


def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw extracted text.
    """
    # Remove non-printable chars
    text = re.sub(r'[^\x00-\x7F\u0900-\u097F]+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove repeated special chars
    text = re.sub(r'[_\-]{3,}', ' ', text)
    # Normalize line breaks
    text = text.replace('\n', ' ').strip()
    return text
