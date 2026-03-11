import cv2
import numpy as np 
from .Constants import *

class DieImageProcessor:
    # Saves the given image to the specified directory with the given filename.
    def SaveImage(self, directory, filename, image):
        path = directory / f"{filename}.png"
        cv2.imwrite(str(path), image)
        return path
    
    def ROI(self, img):
        if img is None:
            return None
        
        # Convert to HSV colour space to detect blue die easier
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define the blue range
        lower_blue = np.array([90, 60, 60])
        upper_blue = np.array([130, 255, 255])

        # Mask only blue areas in the image. This will create a new black and white image, where pixels 
        # falling inside the blue range become white (255), and all other pixels become black (0).
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find the outline of all white shapes in the mask.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #If no contours were found
        if not contours:
            if DEBUG:
                print("No die detected")

            return None

        #Get the largest contour by area, which should be the die
        largest = max(contours, key=cv2.contourArea)

        #Get the bounding box coordinates around the largest contour
        x, y, w, h = cv2.boundingRect(largest)

        #Add Padding to the bounding box to avoid any noise
        paddingX = 20
        paddingY = 20
        

        x1 = max(0, x + paddingX)
        y1 = max(0, y + paddingY)
        x2 = min(img.shape[1], x + w - paddingX)
        y2 = min(img.shape[0], y + h - paddingY)

        if x2 <= x1 or y2 <= y1:
            return None

        #Crop the ROI from the original image using the bounding box coordinates
        #print(f'x:{x}, y:{y}, w:{w}, h:{h}')
        roi = img[y1:y2, x1:x2]

        if DEBUG:
            self.SaveImage(IMAGES_DIR, ROI_IMAGE, roi)

        return roi

    def ProcessImage(self, img):
        if img is None:
            return None
        
        #1. Apply grayscaling to the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #2. Use Gaussian blur to reduce noise and improve OCR accuracy
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        #3. Perform thresholding to get image with only black and white
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #4. Perform morphological operations to clean up the image
        # Define a small 3x3 matrix kernel
        kernel = np.ones((3,3), np.uint8)

        # Remove small white dots (noise)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        if DEBUG:
            #Save image for OCR tool
            self.SaveImage(IMAGES_DIR, PROCESSED_IMAGE, opened)
            
        return opened