import pytesseract
import cv2
import numpy as np
import re
from .Constants import *

class OCRProcessor:
    def __init__(self, tesseract_path=None):
        # Only set path if provided 
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract_text(self, image):
        # processed = self.preprocess_text(image)

        config = r'--oem 3 --psm 6'
        # text = pytesseract.image_to_string(processed, config=config, lang='eng')
        text = pytesseract.image_to_string(image, config=config, lang='eng')

        # Clean text
        text = re.sub(r"[^a-zA-Z0-9\s.,'-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text
        
    def extract_numbers(self, image):
        # config to optimize for digits only and to use the best OCR engine mode
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1'
        
        # Run OCR
        text = pytesseract.image_to_string(image, config=config)

        # Normalize OCR output
        text = text.strip()

        # Keep only digits
        numbers = re.sub(r"[^0-9]", "", text)

        # ---------------- VALIDATION ----------------

        # 1. Empty result
        if not numbers:
            return ""

        # 2. Too many digits (prevents "123", "1020", etc.)
        if len(numbers) > 2:
            return ""

        # 3. Domain constraint (stats range)
        num = int(numbers)
        if num > MAX_POINTS:
            return ""

        return numbers


    def run(self, image, mode=1):
        if image is None:
            raise ValueError("Image is None")

        if mode == 0:
            return self.extract_numbers(image)

        elif mode == 1:
            return self.extract_text(image)

        else:
            raise ValueError("Mode must be 0 (numbers) or 1 (text)")