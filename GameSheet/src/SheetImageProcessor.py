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

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Remove salt-and-pepper noise
        blur = cv2.medianBlur(gray, 3)

        # 3. Perform thresholding to get image with only black background and white text
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

        # 4. Morphological cleanup
        kernel = np.ones((3,3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        if DEBUG:
            self.SaveImage(PROCESSED_DIR, imageName, opened)

        return opened

    # Segments individual digits from a processed binary image and returns them as normalized images.
    def SegmentDigit(self, img, imageName):

        # Detect contours representing potential digits
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        #Get the contours that are most likely the digits
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter small contours that are likely noise
            if h > 20 and w > 5:
                boxes.append((x, y, w, h))

        # Sort detected digits from left to right
        boxes = sorted(boxes, key=lambda b: b[0])

        #Important for double digit cases
        digitImages = []

        for i, (x, y, w, h) in enumerate(boxes):

            # Crop digit region
            digit = img[y:y+h, x:x+w]

            # Normalize digit for CNN input
            digit = self.NormalizeDigit(digit)
            digitImages.append(digit)

            # Save debug image
            if DEBUG:
                self.SaveImage(DEBUG_DIR, f"{imageName}_digit_{i}", digit)

        return digitImages

    # Converts a digit image to a centered 28x28 square suitable for CNN input.
    def NormalizeDigit(self, digitImg):

        # Ensure binary image
        _, digitImg = cv2.threshold(digitImg, 127, 255, cv2.THRESH_BINARY)

        h, w = digitImg.shape

        # Create square canvas based on largest dimension
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)

        # Center digit inside the square
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digitImg

        # Resize to CNN input size
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

        return resized

    # def NormalizeDigit(self, digitImg):

    #     # Ensure binary
    #     _, digitImg = cv2.threshold(digitImg, 127, 255, cv2.THRESH_BINARY)

    #     # Find bounding box of digit pixels
    #     coords = cv2.findNonZero(digitImg)
    #     x, y, w, h = cv2.boundingRect(coords)
    #     digit = digitImg[y:y+h, x:x+w]

    #     # Resize while preserving aspect ratio
    #     target_size = 20
    #     h, w = digit.shape

    #     if h > w:
    #         new_h = target_size
    #         new_w = int(w * (target_size / h))
    #     else:
    #         new_w = target_size
    #         new_h = int(h * (target_size / w))

    #     resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    #     # Create 28x28 canvas
    #     canvas = np.zeros((28, 28), dtype=np.uint8)

    #     y_offset = (28 - new_h) // 2
    #     x_offset = (28 - new_w) // 2

    #     canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    #     return canvas