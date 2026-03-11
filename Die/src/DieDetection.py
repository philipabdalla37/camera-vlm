import cv2
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

        # Camera will be initialized later
        self.cap = None

    #Get the current frame
    def GetFrame(self):

        # Capture frame-by-frame
        if DEBUG_CAMO:
            #Use video capture to get the frame
            ret, frame = self.cap.read()
            if not ret:
                return None
            
        #Use PiCamera to get the frame
        else:
            frame = self.cap.capture_array()

            # Convert RGB → BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    # Detects if a die is present using colour segmentation.
    # Assumes high-contrast die (e.g., dark die on light surface).
    def DieInFrame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue detection range
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

    #Show Debug Window
    def ShowDebugFrame(self, frame):

        display = cv2.resize(frame, DISPLAY_SIZE)
        cv2.imshow("Camera Capture", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

    #Take a photo of the frame and run Image Processing
    def CaptureFrame(self):

        if self.curFrame is None:
            return

        if DEBUG:
            self.processor.SaveImage(IMAGES_DIR, ORIGINAL_IMAGE, self.curFrame)
            print("Image saved")

        self.runImageProcessing()

    # Process the captured image and get die number
    def runImageProcessing(self):

        roi = self.processor.ROI(self.curFrame)
        processed = self.processor.ProcessImage(roi)

        #Teachable Machine selected
        if self.curMethod == self.method["CNN"]:

            result, confidence = self.model.TeachableMachine(processed)

            self.dieResult = result

            if DEBUG:
                print("Result:", result)
                print("Confidence:", confidence)

    #Main detection loop that won't stop until die is detected and output is sent
    def DetectionLoop(self):

        while True:

            frame = self.GetFrame()

            if frame is None:
                continue

            #Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Get the most recent frame
            self.curFrame = frame.copy()

#######################################################################
                     # DIE PRESENCE CHECK #
#######################################################################

            diePresent = self.DieInFrame(frame)
            
            # Mark the die as present
            if diePresent:

                if not self.dieDetected:
                    self.dieDetected = True
                    self.prevGray = gray
                    self.stillFrames = 0

            else:
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
                    print("Frame movement: ", movement)

                #If this movement is less than the required one, increment the still flag. 
                #This will allow the program to see how many consecutive frames the die has been still.
                if movement < MOVEMENT_THRESHOLD:
                    self.stillFrames += 1

                    if DEBUG:
                        print("stillFrame = ", self.stillFrames)
                
                else:
                    self.stillFrames = 0

                    if DEBUG:
                        print("Motion detected, reset frame counter.")

                #Once we pass the stillness required, take the photo and analyze it. Reset other values
                if (self.stillFrames > STILLNESS_REQUIRED):
                    if DEBUG:
                        print("Capture triggered")
                    
                    self.CaptureFrame()
                    break

                #Set the current frame as the previous one
                self.prevGray = gray

            if DEBUG:
                if not self.ShowDebugFrame(frame):
                    break

    # Main Execution
    def RunDie(self):

        # Initialize video capture
        if DEBUG_CAMO:
            # Force higher resolution
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # Initialize PiCamera2 capture
        else:
            self.cap = Picamera2()
            self.cap.configure(self.cap.create_preview_configuration(main={"size": (1920, 1080)}))
            self.cap.start()

        self.DetectionLoop()

        # Camera cleanup
        if DEBUG_CAMO:

            self.cap.release()

        else:

            self.cap.stop()
            self.cap.close()

        cv2.destroyAllWindows()

        return self.dieResult
    
# die = DieDetection()
# die.runDie()