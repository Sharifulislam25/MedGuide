"""
ocr.py

Runs Tesseract OCR on an already-preprocessed image. Kept separate from
image_processor.py so OCR logic (and its error handling) is easy to
find and change independently of the preprocessing steps.
"""

import pytesseract
from PIL import Image

# --- Windows note ---
# If Tesseract isn't on your system PATH, pytesseract won't find it even
# though it's installed. If you hit a "TesseractNotFoundError" below,
# uncomment the next line and point it at your install location:
#
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def run_ocr(image: Image.Image) -> str:
    """
    Extract text from an image using Tesseract.

    Parameters
    ----------
    image : PIL.Image.Image
        The (ideally preprocessed) image to run OCR on.

    Returns
    -------
    str
        The extracted text, with leading/trailing whitespace trimmed.
        Returns an empty string if Tesseract finds no text at all.
    """
    try:
        text = pytesseract.image_to_string(image)
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
