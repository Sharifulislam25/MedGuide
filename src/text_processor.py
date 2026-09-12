"""
text_processor.py

Cleans raw extracted text (from PDFs, OCR, or plain files) so it's more
consistent for later steps like chunking and embedding.

This module intentionally does NOT touch anything that matters for
medical accuracy: numbers, units, medical terms, reference ranges,
dates, and table values are all preserved exactly as extracted. It only
tidies up formatting noise around that content.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean up common formatting issues in extracted text.

    Parameters
    ----------
    text : str
        Raw text straight out of a PDF, OCR, or a text file.

    Returns
    -------
    str
        Cleaned text: consistent line breaks, no excessive blank
        lines/spaces, and a few common OCR noise characters removed.
    """
    if not text:
        return ""

    # Normalize different line-ending styles (Windows/Mac/Unix) to "\n".
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ blank lines into a single blank line between paragraphs,
    # instead of leaving large empty gaps (common in OCR output).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces/tabs (but never newlines) into one space.
    # OCR text especially tends to have irregular spacing between words.
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove short runs of stray OCR noise characters (table-border or
    # scan-artifact characters like "|||", "~~~", or repeated backticks)
    # that carry no real information. Kept intentionally conservative
    # (requires 2+ in a row) so real content is never touched.
    text = re.sub(r"[|~`]{2,}", "", text)

    # Strip a lone leading punctuation character on a line (a common OCR
    # artifact from stray marks/shadows at the start of a line, e.g.
    # ": Some real content" or ". Some real content"), but only when
    # it's clearly not part of the content itself.
    text = re.sub(r"^[:;.]\s+", "", text, flags=re.MULTILINE)

    # Strip trailing whitespace from each line, then trim the whole text.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    return text
