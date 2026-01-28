import pytesseract
from Constants import *

#Run the given image through pyTesseract (by default PROCESSED_IMAGE)
def runPyTesseract(image=PROCESSED_IMAGE):
    # Path to Tesseract executable (Windows)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Read the cropped die image. oem 3 = Automatically choose best OCR engine. psm 10 = single character recognition
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456'
    text = pytesseract.image_to_string(image, config=custom_config)
    print(f"Detected digit based on {image}:", text.strip())
    return (f"Detected digit based on {image}:", text.strip())
