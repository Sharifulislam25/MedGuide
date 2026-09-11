"""
image_processor.py

Prepares an uploaded image for OCR, and ties preprocessing + OCR
together into a single load_image() function that returns a Document
(same pattern as load_txt() and load_pdf()).

Preprocessing never modifies the original uploaded image — it always
works on a copy, producing a new image just for Tesseract to read.
"""

import io
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2

from src.document_loader import Document
from src.ocr import run_ocr


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Apply a simple, beginner-friendly preprocessing pipeline to improve
    OCR accuracy:
      1. Resize if the image is very large (speeds up OCR without hurting accuracy)
      2. Convert to grayscale
      3. Boost contrast
      4. Threshold to pure black/white so text stands out from the background

    Parameters
    ----------
    pil_image : PIL.Image.Image
        The original uploaded image (left untouched).

    Returns
    -------
    PIL.Image.Image
        A new, processed image ready for OCR.
    """
    # Work on a copy so the original is never modified.
    image = pil_image.copy()

    # 1. Resize if very large.
    max_dimension = 2000
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size)

    # 2. Grayscale — OCR works on light/dark contrast, not color.
    image = ImageOps.grayscale(image)

    # 3. Contrast enhancement — makes faint text easier for Tesseract to catch.
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    # 4. Thresholding (via OpenCV) — converts to pure black/white using
    #    Otsu's method, which picks a good cutoff automatically instead
    #    of us having to guess one.
    image_array = np.array(image)
    _, thresholded = cv2.threshold(
        image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    image = Image.fromarray(thresholded)

    return image


def load_image(uploaded_file) -> Document:
    """
    Read an uploaded image, preprocess it, run OCR, and return the
    result as a standardized Document.

    Parameters
    ----------
    uploaded_file : a Streamlit UploadedFile object

    Returns
    -------
    Document
        Contains the OCR-extracted text and ocr=True in its metadata.
    """
    pil_image = Image.open(io.BytesIO(uploaded_file.read()))
    processed_image = preprocess_image(pil_image)
    extracted_text = run_ocr(processed_image)

    return Document(
        text=extracted_text,
        source=uploaded_file.name,
        file_type="image",
        page=1,
        metadata={"ocr": True}
    )
