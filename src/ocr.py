"""
ocr.py

Runs Tesseract OCR on an already-preprocessed image. Kept separate from
image_processor.py so OCR logic (and its error handling) is easy to
find and change independently of the preprocessing steps.
"""

import pytesseract
from PIL import Image

import os

# Set Tesseract binary path for Windows if present
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# --psm 6 tells Tesseract to "assume a single uniform block of text".
# For scanned forms/reports (as opposed to a photo of a single line, or
# a page with sparse scattered text), this reads noticeably better than
# Tesseract's default page-segmentation mode.
DEFAULT_OCR_CONFIG = "--psm 6"


def run_ocr(image: Image.Image, config: str = DEFAULT_OCR_CONFIG) -> str:
    """
    Extract text from an image using Tesseract.

    Parameters
    ----------
    image : PIL.Image.Image
        The (ideally preprocessed) image to run OCR on.
    config : str
        Tesseract command-line options. Defaults to "--psm 6". If OCR
        results look wrong for a particular kind of document, this is
        worth experimenting with — e.g. "--psm 4" tends to work better
        for text laid out in columns.

    Returns
    -------
    str
        The extracted text, with leading/trailing whitespace trimmed.
        Returns an empty string if Tesseract finds no text at all.
    """
    try:
        text = pytesseract.image_to_string(image, config=config)
    except pytesseract.TesseractNotFoundError:
        # Turn the cryptic library error into a clear message that
        # points the user at the fix.
        raise RuntimeError(
            "Tesseract OCR engine was not found on this computer. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "(Windows build) and make sure it's on your system PATH, or set "
            "pytesseract.pytesseract.tesseract_cmd in src/ocr.py."
        )

    return text.strip()
