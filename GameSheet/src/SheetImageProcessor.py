from pathlib import Path

import cv2
import numpy as np
import os
from .Constants import *

class SheetImageProcessor:

    # Saves the given image to the specified directory with the given filename.
    def SaveImage(self, directory, filename, image):
        path = directory / f"{filename}.png"
        cv2.imwrite(str(path), image)
        return path
    
    #Get the Region of Interest (ROI) of the sheet section.
    def ROI(self, img, imageName):

        paddingX1, paddingY1, paddingX2, paddingY2 = ROI_PADDING

        h, w = img.shape[:2]
        roi = img[paddingY1:(h-paddingY2), paddingX1:(w-paddingX2)]

        # Increase image size for better segmentation
        clean = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        if DEBUG:
            self.SaveImage(ROI_DIR, imageName, clean)

        return clean

    # Image Processing Pipeline

    def ProcessImage(self, img, imageName):

        # 1. Ensure grayscale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Light denoise (preserve edges)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3. Global threshold
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4. Light morphological cleanup to avoid over-thickening
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        if DEBUG:
            self.SaveImage(PROCESSED_DIR, imageName, cleaned)

        return cleaned