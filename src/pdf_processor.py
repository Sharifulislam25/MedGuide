"""
pdf_processor.py

Extracts text from PDF files using PyMuPDF (imported as "fitz").

For now (Phase 3), this only handles normal PDFs that already contain
selectable text. Scanned PDFs with little or no extractable text will
get an OCR fallback in Phase 5 — this module will be extended then,
not replaced.
"""

import fitz  # PyMuPDF
from typing import List
from src.document_loader import Document


def load_pdf(uploaded_file) -> List[Document]:
    """
    Read a PDF uploaded through Streamlit's file_uploader and return one
    Document per page.
    """
    pdf_bytes = uploaded_file.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    documents = []
    for page_number, page in enumerate(pdf, start=1):
        page_text = page.get_text()

        documents.append(
            Document(
                text=page_text,
                source=uploaded_file.name,
                file_type="pdf",
                page=page_number,
                metadata={"ocr": False}
            )
        )

    pdf.close()
    return documents
    