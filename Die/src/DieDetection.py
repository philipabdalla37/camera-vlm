import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

from .DieImageProcessor import DieImageProcessor
from .Teachable import Teachable
from .Constants import *

if not DEBUG_CAMO:
    from picamera2 import Picamera2

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

        # Initialize processors
        self.processor = DieImageProcessor()
        self.model = Teachable()

        #Set current method 
        self.curMethod = self.method["CNN"]

        # Initialize video capture
        if DEBUG_CAMO:
            self.cap = cv2.VideoCapture(0)

            # Force higher resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # Initialize PiCamera2 capture
        else:
            self.cap = Picamera2()
            self.cap.configure(self.cap.create_preview_configuration(main={"size": (1920, 1080)}))
            self.cap.start()

        if DEBUG:
            # Tkinter GUI
            self.root = tk.Tk()
            self.root.title("Camera Capture")

            # Label to show the camera frame
            self.videoLabel = ttk.Label(self.root)
            self.videoLabel.pack()

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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Example range (adjust if needed)
        lower = np.array([90, 60, 60])
        upper = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        #Evaluate the largest contour, and if it's too small, ignore it as noise
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < CONTOUR_AREA_THRESHOLD:
            return False


        
        if DEBUG:
            # Draw bounding box LIVE
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return True

######################################################################
                            # FRAME LOOP #
######################################################################
    def showFrame(self):
        if self.isPaused:
            return
            
        # Capture frame-by-frame
        if DEBUG_CAMO:
            #Use video capture to get the frame
            ret, frame = self.cap.read()
            if not ret:
                return

        #Use PiCamera to get the frame
        else:
            frame = self.cap.capture_array()

            # Convert RGB → BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Get the most recent frame
        self.curFrame = frame.copy()

#######################################################################
                     # DIE PRESENCE CHECK #
#######################################################################
        diePresent = self.dieInFrame(frame)
        
        # Mark the die as present
        if diePresent:
            if DEBUG:
                print("1. Die is in the frame")

            if not self.dieDetected:
                self.dieDetected = True
                self.prevGray = gray
                self.stillFrames = 0
                if DEBUG:
                    print("2a. Die detected changed to true")
        else:
            if DEBUG and self.dieDetected:
                print("2b. Die detected changed to false")

            self.dieDetected = False
            
#########################################################################
                        # STILLNESS CHECK #
#########################################################################
        # Motion detection: If the die is present, check motion to see when it stops
        if self.dieDetected:
            
            #Compare the difference between the previous frame and the new one.
            diff = cv2.absdiff(gray, self.prevGray)
            
            #If movement is close to 0, it means the image is almost identical,
            #so the die is not moving anymore.
            movement = np.sum(diff) / diff.size

            if DEBUG:
                print("4. The movement is: ", movement)

            #If this movement is less than the required one, increment the still flag. 
            #This will allow the program to see how many consecutive frames the die has been still.
            if movement < MOVEMENT_THRESHOLD:
                self.stillFrames += 1

                if DEBUG:
                    print("5. stillFrame = ", self.stillFrames)
                
            else:
                self.stillFrames = 0

                if DEBUG:
                    print("6. Motion detected, reset")

            #Once we pass the stillness required, take the photo and analyze it. Reset other values
            if (self.stillFrames > STILLNESS_REQUIRED and AUTO_CAPTURE):
                if DEBUG:
                    print("7. Capture triggered")
                
                self.onCaptureFrame()
                self.dieDetected = False
                self.stillFrames = 0

            #Set the current frame as the previous one
            self.prevGray = gray

#######################################################################
                        # DISPLAY FRAME #
#######################################################################
        if DEBUG:
            display = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)

            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(display_rgb)
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
        if self.curFrame is None:
            return

        if DEBUG:
            self.resumeButton.pack(pady=10)  # Shows the resume button
            self.processor.SaveImage(IMAGES_DIR, ORIGINAL_IMAGE, self.curFrame)
            print("Image saved")

        self.isPaused = True            
        
        self.runImageProcessing()

    # Function to process the captured image
    def runImageProcessing(self):
        roi = self.processor.ROI(self.curFrame)
        processed = self.processor.ProcessImage(roi)

        #Teachable Machine selected
        if self.curMethod == self.method["CNN"]:
            result, confidence = self.model.TeachableMachine(processed)

            self.dieResult = result

            #Stop the GUI loop
            if DEBUG:
                print("Result:", result)
                print("Confidence:", confidence)
                self.root.quit()

    def runDie(self):

        # Start the video loop
        self.showFrame()

        # Runs app
        if DEBUG:
            self.root.mainloop()

        # Cleanup after window is closed
        if DEBUG_CAMO:
            #Stop video capture.
            self.cap.release()
    
        else:
            #Stop PiCamera
            self.cap.stop()

        cv2.destroyAllWindows()

        return self.dieResult

# die = DieDetection()
# die.runDie()