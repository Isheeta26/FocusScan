import easyocr
import numpy as np
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can specify multiple languages if needed

def extract_text(image, boxes):
    img = Image.open(image).convert("RGB")
    ocr_texts = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = img.crop((x1, y1, x2, y2))  # Crop the region of interest

        # Perform OCR to extract text
        ocr_results = reader.readtext(np.array(roi))
        ocr_text = " ".join([result[1] for result in ocr_results]) if ocr_results else "No text detected"

        ocr_texts.append(ocr_text)

    return ocr_texts
