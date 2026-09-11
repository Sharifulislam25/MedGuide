"""
pdf_processor.py

Extracts text from PDF files using PyMuPDF (imported as "fitz").

Normal pages (with a real text layer) are read directly. Scanned pages
(little or no extractable text) are rendered as images and run through
the same OCR pipeline used for uploaded images in Phase 4.
"""

import io
import fitz  # PyMuPDF
from typing import List
from PIL import Image

from src.document_loader import Document
from src.image_processor import preprocess_image
from src.ocr import run_ocr

# If a page has fewer than this many extractable characters, we treat it
# as "scanned" (no usable text layer) and fall back to OCR instead.
MIN_TEXT_LENGTH_FOR_NORMAL_PDF = 20


def load_pdf(uploaded_file) -> List[Document]:
    """
    Read a PDF uploaded through Streamlit's file_uploader and return one
    Document per page. Pages with a real text layer are read directly;
    pages without one are rendered as images and OCR'd automatically.

    Parameters
    ----------
    uploaded_file : a Streamlit UploadedFile object

    Returns
    -------
    List[Document]
        One Document per page. Each Document's metadata["ocr"] tells you
        whether that specific page needed OCR.
    """
    pdf_bytes = uploaded_file.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    documents = []
    for page_number, page in enumerate(pdf, start=1):
        page_text = page.get_text()

        if len(page_text.strip()) >= MIN_TEXT_LENGTH_FOR_NORMAL_PDF:
            # Normal page: the text layer already gives us usable text.
            documents.append(
                Document(
                    text=page_text,
                    source=uploaded_file.name,
                    file_type="pdf",
                    page=page_number,
                    metadata={"ocr": False}
                )
            )
        else:
            # Likely a scanned page: render it as a high-resolution image,
            # then run it through the same preprocessing + OCR pipeline
            # used for uploaded images (src/image_processor.py, src/ocr.py).
            pixmap = page.get_pixmap(dpi=300)
            image_bytes = pixmap.tobytes("png")
            pil_image = Image.open(io.BytesIO(image_bytes))

            processed_image = preprocess_image(pil_image)
            ocr_text = run_ocr(processed_image)

            documents.append(
                Document(
                    text=ocr_text,
                    source=uploaded_file.name,
                    file_type="pdf",
                    page=page_number,
                    metadata={"ocr": True}
                )
            )

    pdf.close()
    return documents
