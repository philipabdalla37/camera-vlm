import cv2
import numpy as np
import os
from Constants import *

def ROI(image, imageName):

    # Read original image
    img = cv2.imread(image)

    #Add Padding to the bounding box to avoid any noise
    paddingX1, paddingY1, paddingX2, paddingY2 = ROI_PADDING
    
    h,w = img.shape[:2]
    roi = img[paddingY1:(h-paddingY2), paddingX1:(w-paddingX2)]

    #Increase image size
    clean = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(f"images/roi/{imageName}.png", clean)

def ProcessImage(image):

    # Get image name
    imageName = os.path.basename(image)          # e.g. "sheet1.png"

    #1. Apply grayscaling to the image
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #2. Use meadian blur to reduce salt and pepper noise
    blur = cv2.medianBlur(gray, 3)

    #3. Perform thresholding to get image with only black and white
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

    #4. Perform morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)

    # Remove small white dots (noise)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, 1)

    #Save image for OCR tool
    cv2.imwrite(f"images/processedSections/{imageName}", closed)