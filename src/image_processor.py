"""
image_processor.py

Prepares an uploaded image for OCR, and ties preprocessing + OCR
together into a single load_image() function that returns a Document
(same pattern as load_txt() and load_pdf()).

Preprocessing never modifies the original uploaded image — it always
works on a copy, producing a new image just for Tesseract to read.
"""

import io
from PIL import Image, ImageOps
import numpy as np
import cv2

from src.document_loader import Document
from src.ocr import run_ocr

# If an uploaded image is narrower than this, we upscale it before OCR.
# Small/compressed photos (e.g. a 600x730 phone photo of a form) tend to
# have text too small for Tesseract to read reliably otherwise.
MIN_WIDTH_FOR_OCR = 1800

# Safety cap so we don't blow up memory on an unusually huge upload.
MAX_WIDTH_FOR_OCR = 3000


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Apply a preprocessing pipeline to improve OCR accuracy:
      1. Upscale if the image is small (most common cause of poor OCR
         on phone photos / low-res scans); downscale if it's huge.
      2. Convert to grayscale.
      3. Sharpen — counters the blur introduced by upscaling and by
         JPEG compression.
      4. Adaptive thresholding — converts to black/white using a
         locally-computed cutoff, which handles the uneven lighting
         and shadows typical of a photographed (not flatbed-scanned)
         document better than a single global threshold.

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

    # 1. Resize: upscale small images, downscale very large ones.
    #    INTER_LANCZOS4 is a higher-quality (if slower) resize algorithm
    #    than the default — it matters here because we're stretching a
    #    small image up rather than just shrinking a large one.
    width, height = image.size
    if width < MIN_WIDTH_FOR_OCR:
        scale = MIN_WIDTH_FOR_OCR / width
    elif width > MAX_WIDTH_FOR_OCR:
        scale = MAX_WIDTH_FOR_OCR / width
    else:
        scale = 1.0

    if scale != 1.0:
        new_size = (int(width * scale), int(height * scale))
        image_array = np.array(image)
        image_array = cv2.resize(image_array, new_size, interpolation=cv2.INTER_LANCZOS4)
        image = Image.fromarray(image_array)

    # 2. Grayscale — OCR cares about light/dark contrast, not color.
    image = ImageOps.grayscale(image)
    gray = np.array(image)

    # 3. Sharpen using an "unsharp mask": blur a copy, then subtract a
    #    weighted amount of that blur from the original. This makes
    #    letter edges crisper, which helps a lot after upscaling.
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # 4. Adaptive thresholding instead of a single global cutoff (Otsu).
    #    A photographed page often has uneven lighting — one corner
    #    brighter than another — and a single global threshold can turn
    #    the darker side into a black blob. Adaptive thresholding
    #    computes a local cutoff for each neighborhood instead.
    thresholded = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

    return Image.fromarray(thresholded)


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
