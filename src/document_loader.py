"""
document_loader.py

Turns an uploaded file into a standardized "Document" object, no matter
what file type it came from. In later phases this module will also
handle PDF, image, and Markdown files. For now (Phase 2), it only knows
how to read plain .txt files.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """
    A standardized representation of any loaded document.

    Every file type (PDF, image, TXT, MD) will eventually be converted
    into this same shape, so the rest of the app never needs to know or
    care where the text originally came from.
    """
    text: str
    source: str                # original file name
    file_type: str             # "txt", "pdf", "image", "md"
    page: Optional[int] = None
    metadata: dict = field(default_factory=dict)


def load_txt(uploaded_file) -> Document:
    """
    Read a plain .txt file uploaded through Streamlit's file_uploader.

    Parameters
    ----------
    uploaded_file : a Streamlit UploadedFile object

    Returns
    -------
    Document
        A standardized Document containing the file's text.
    """
    # Streamlit gives us the file as raw bytes; decode into a normal string.
    # errors="replace" prevents a crash if the file has a few odd characters.
    raw_bytes = uploaded_file.read()
    text = raw_bytes.decode("utf-8", errors="replace")

    return Document(
        text=text,
        source=uploaded_file.name,
        file_type="txt",
        page=1,
        metadata={"ocr": False}
    )
