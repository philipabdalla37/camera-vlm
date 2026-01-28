import cv2
import numpy as np 
from Constants import *

def ROI(image):
    # Read original image
    img = cv2.imread(image)
    
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

    #If any contours were found
    if contours:
        #Get the largest contour by area, which should be the die
        largest = max(contours, key=cv2.contourArea)

        #Get the bounding box coordinates around the largest contour
        x, y, w, h = cv2.boundingRect(largest)

        #Add Padding to the bounding box to avoid any noise
        paddingX = 20
        paddingY = 20
        
        #Crop the ROI from the original image using the bounding box coordinates
        #print(f'x:{x}, y:{y}, w:{w}, h:{h}')
        roi = img[y+paddingY:y+(h-paddingY), x+paddingX:x+(w-paddingX)]

        #Save the cropped image as IMAGE 
        cv2.imwrite(ROI_IMAGE, roi)

    else:
        print("No die detected.")

def ProcessImage(image):
    #1. Apply grayscaling to the image
    img = cv2.imread(image)
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

    #Save image for OCR tool
    cv2.imwrite(PROCESSED_IMAGE, opened)
    