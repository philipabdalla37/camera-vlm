import cv2
import os
import shutil
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from picamera2 import Picamera2, Preview

from .ImageProcessing import *
from .Teachable import *

DISPLAY_SIZE = (640, 360)
MOVEMENT_THRESHOLD = 6.0      # Adjust if needed
STILLNESS_REQUIRED = 10       # Frames of stillness before capture
CONTOUR_AREA_THRESHOLD = 1500
AUTO_CAPTURE = True
DEBUG = True

class DieDetection:
    
    #Different methods for die recognition
    method = {
        "OCR": 1,
        "CNN": 2,
        "VLM": 3
    }

    def __init__(self):
        self.isPaused = False
        self.curFrame = None
        self.prevGray = None
        self.stillFrames = 0
        self.dieDetected = False
        self.dieResult = None

        #Set current method 
        self.curMethod = self.method["CNN"]

        # Initialize PiCamera2 capture
        self.cap = Picamera2()
        self.cap.configure(self.cap.create_preview_configuration(main={"size": (1920, 1080)}))
        self.cap.start()

        # Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Camera Capture")

        # Label to show the camera frame
        self.videoLabel = ttk.Label(self.root)
        self.videoLabel.pack()

        if DEBUG:
            self.screenshotButton = tk.Button(
                self.root,
                text="Capture Frame",
                command=self.onCaptureFrame
            )
            self.screenshotButton.pack(pady=10)

            self.resumeButton = tk.Button(
                self.root,
                text="Resume Camera",
                command=self.onResume
            )

    # Detects if a die is present using colour segmentation.
    # Assumes high-contrast die (e.g., dark die on light surface).
    def dieInFrame(self, frame):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Example range (adjust if needed)
        lower = np.array([90, 60, 60])
        upper = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        #If no die exists in the frame yet
        if len(contours) == 0:
            return False
        
        #Evaluate the largest contour, and if it's too small, ignore it as noise
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < CONTOUR_AREA_THRESHOLD:
            return False

        x, y, w, h = cv2.boundingRect(largest)
        
        if DEBUG:
            # Draw bounding box LIVE
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return True

    # Function to show camera frame
    def showFrame(self):
        if not self.isPaused:
            frame = self.cap.capture_array()

            #Get the most recent frame
            self.curFrame = frame.copy()

            # Die detection: See if the die is visible in the frame, and mark it as detected
            if self.dieInFrame(frame):
                if DEBUG:
                    print("1. Die is in the frame")

                if not self.dieDetected:
                    self.dieDetected = True
                    self.prevGray = None
                    self.stillFrames = 0
                    if DEBUG:
                        print("2. Die detected changed to true")
            else:
                self.dieDetected = False
                if DEBUG:
                    print("3. Die detected changed to false")

            # Motion detection: If the die is present, check motion to see when it stops
            if self.dieDetected:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                if DEBUG:
                    print("4. Die detected true, so now the magic happens")

                if self.prevGray is not None:
                    #Compare the difference between the previous frame and the new one.
                    diff = cv2.absdiff(gray, self.prevGray)
                    if DEBUG:
                        print("5. The difference between the last two frames is: ", diff)

                    #If movement is close to 0, it means the image is almost identical,
                    #so the die is not moving anymore.
                    movement = np.sum(diff) / diff.size

                if DEBUG:
                    print("6. The movement is: ", movement)

                    #If this movement is less than the required one, increment the still flag. 
                    #This will allow the program to see how many consecutive frames the die has been still.
                    if movement < MOVEMENT_THRESHOLD:
                        if DEBUG:
                            print("7. Movement is less than threshold, thus add stillFrame = ", self.stillFrames)
                        self.stillFrames += 1
                    else:
                        self.stillFrames = 0

                    #Once we pass the stillness required, take the photo and analyze it. Reset other values
                    if (self.stillFrames > STILLNESS_REQUIRED and AUTO_CAPTURE):
                        self.onCaptureFrame()
                        self.dieDetected = False
                        self.stillFrames = 0

                #Set the current frame as the previous one
                self.prevGray = gray
            
            #Display the frame
            display = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
            img = Image.fromarray(display)
            imgtk = ImageTk.PhotoImage(image=img)

            #Update the label with this image
            self.videoLabel.imgtk = imgtk
            self.videoLabel.configure(image=imgtk)

            #Wait 10ms, then call this function again so that the frames keep updating
            self.videoLabel.after(10, self.showFrame)

    # Function to resume camera feed
    def onResume(self):
        self.resumeButton.pack_forget()  # Hides the resume button
        self.isPaused = False
        self.showFrame()                 # Resumes the camera feed

    # Function to capture and save frame
    def onCaptureFrame(self):
        if self.curFrame is not None:

            self.resumeButton.pack(pady=10)  # Shows the resume button
            self.isPaused = True

            #Modify format from RGB -> BGR, for cv2 to process the image
            bgr = cv2.cvtColor(self.curFrame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ORIGINAL_IMAGE, bgr)

            print("Image saved as snapshot.jpg")
            
            self.runImageProcessing(ORIGINAL_IMAGE)

    # Function to process the captured image
    def runImageProcessing(self, image):
        ROI(image)
        ProcessImage(ROI_IMAGE)

        #Teachable Machine selected
        if self.curMethod == self.method["CNN"]:
            output = TeachableMachine(PROCESSED_IMAGE)
            
            # messagebox.showinfo("Result", output)

            self.dieResult = output

            #Stop the GUI loop
            self.root.quit()

    def runDie(self):

        # Start the video loop
        self.showFrame()

        # Runs app
        self.root.mainloop()

        # Cleanup after window is closed
        self.cap.stop()
        cv2.destroyAllWindows()

        return self.dieResult

# die = DieDetection()
# die.runDie()