import cv2
import os
import time
import json
import numpy as np
from pathlib import Path

from .Constants import *
from .ImageProcessor import ImageProcessor
from .DigitRecognizer import DigitRecognizer
# from picamera2 import Picamera2

class TextDetection:

    def __init__(self):
        # Ensure directories exist
        if DEBUG:
            for directory in ALL_DIRS:
                directory.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.processor = ImageProcessor()

        #Get the CNN Model
        self.digitRecognizer = DigitRecognizer(DIGIT_H5)

        if DEBUG_CAMO:
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)

            # Force higher resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        else:
            # Initialize PiCamera2
            self.cap = Picamera2()
            self.cap.configure(
                self.cap.create_preview_configuration(
                    main={"size": (1920, 1080)}
                )
            )
            self.cap.start()


    # Generates a padded ArUco marker for the given ID and saves it as a PNG file.
    def GenerateArucoMarkers(self, markerId, aruco_type=cv2.aruco.DICT_6X6_250, marker_size=800, border_bits=1):
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
        marker = cv2.aruco.generateImageMarker(dictionary, markerId, marker_size, borderBits=border_bits)
        
        # Add external white padding
        margin = int(marker_size * 0.25)
        padded_size = marker_size + 2 * margin
        padded = 255 * np.ones((padded_size, padded_size), dtype=np.uint8)
        padded[margin:margin + marker_size, margin:margin + marker_size] = marker
        self.processor.SaveImage(MARKERS_DIR, f"aruco_marker_{markerId}_padded", padded)

    #Crops and rectifies the sheet from the camera frame using the detected corner points.
    def PerspectiveTransform(self, centers, frame):

        #Define size of final cropped image
        TARGET_WIDTH = 584
        TARGET_HEIGHT = 567

        #Take the detected corners and map them to the destination ones using Perspective transform.
        sourcePoints = np.array([
            centers[0],  # top-left
            centers[1],  # top-right
            centers[3],  # bottom-right
            centers[2]   # bottom-left
        ], dtype=np.float32)

        # Destination points 
        destPoints = np.array([
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1]
        ], dtype=np.float32)

        # Compute perspective transform from detected corners to target rectangle
        matrix = cv2.getPerspectiveTransform(sourcePoints, destPoints)

        # Apply transform to obtain a top-down warped image of the sheet
        warped = cv2.warpPerspective(frame, matrix, (TARGET_WIDTH, TARGET_HEIGHT))

        return warped

    # Extracts a specific section from the cropped sheet image based on predefined coordinates and saves it as a PNG file.
    def extractSection(self, img, section):

        #Get section name and coordinates
        name = section[1]
        x1, y1, x2, y2 = section[2]

        #Crop the image based on section
        sectionImage = img[y1:y2, x1:x2]
        
        # Save the section as a PNG file
        if DEBUG:
            self.processor.SaveImage(SECTIONS_DIR, name, sectionImage)

        return sectionImage

    # Detects ArUco markers in the frame and returns their centers and detection data.
    def DetectMarkers(self, aruco_type=cv2.aruco.DICT_6X6_250):

        print(JSON_DIR)
        #Loads trained CNN model for digit recognition
        digitRecognizer = DigitRecognizer("camera-vlm/GameSheet/models/digit_model.h5")
        
        # Each marker contains a 6×6 binary grid, which allows for a total of 250 unique markers in the DICT_6X6_250 dictionary.
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)

        # Creates a parameter object controlling detection behavior.
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        #Detects the markers in the video feed.
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        # Flags and variables to check orientation and stillness
        orientationLocked = False
        prevGray = None
        stillFrames = 0

        #Timer to check for no sheet detection
        startTime = time.time()
        
        # Main loop to process video frames
        while True:
            # Stop if no sheet detected within TIMEOUT
            if time.time() - startTime > TIMEOUT:
                if DEBUG:
                    print("Timeout: No sheet detected within 15 seconds.")
    
                return False

            if DEBUG:
                elapsed = time.time() - startTime
                remaining = TIMEOUT - elapsed
                print(f"Time remaining: {remaining:.1f} seconds")

            # Capture frame-by-frame
            if DEBUG_CAMO:
                #Use video capture to get the frame
                ret, frame = self.cap.read()
                if not ret:
                    break

            else:
                #Use PiCamera to get the frame
                frame = self.cap.capture_array()

                # Convert RGB → BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            #Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Detect markers in the current frame. The function returns the corners of the detected markers, their IDs, and any rejected candidates.
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)

            # Store the centers of detected markers
            centers = {}
            
###############################################################################################################
                                         # DETECT MARKERS #
###############################################################################################################
            if marker_ids is not None:
                #Convert to 1D array for easier processing
                marker_ids = marker_ids.flatten()
                startTime = time.time()        # reset timer
                
                if DEBUG:
                    print("Detected IDs:", marker_ids)

                #Compute the center of each marker. zip() allows to associate each set of corners with its corresponding marker ID.
                for corners, markerId in zip(marker_corners, marker_ids):
                    
                    #Get the first point set of the given marker, calculate its center, and store it 
                    pts = corners[0]
                    center = np.mean(pts, axis=0)
                    centers[markerId] = center

                #Draws a green bounding box around each detected marker and labels it with its ID.
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

###############################################################################################################
                                        # ORIENTATION CHECK #
###############################################################################################################
            #Check for the presence of all 4 markers, and if they are in the correct orientation.
            isOrientationCorrect = False

            #Check that all 4 markers are detected
            if set(centers.keys()) == ARUCO_IDS:
                c0 = centers[0]     #(top-left)
                c1 = centers[1]     #(top-right)
                c2 = centers[2]     #(bottom-left)

                #Get the top and left vector
                v_top = c1 - c0
                v_left = c2 - c0

                #Get the angles of the top and left vectors
                angle_top = np.degrees(np.arctan2(v_top[1], v_top[0]))
                angle_left = np.degrees(np.arctan2(v_left[1], v_left[0]))

                if DEBUG:
                    print("Angles: top =", angle_top, ", left =", angle_left)
                
                #Controls correct orientation
                cond_top = abs(angle_top) < ANGLE_TOLERANCE
                cond_left = abs(abs(angle_left) - 90) < ANGLE_TOLERANCE
            
                #Correct orientation if the angle condition is met for both vectors
                if cond_top and cond_left:
                    isOrientationCorrect = True

###############################################################################################################
                                        # STATE TRANSITIONS #
###############################################################################################################
            #After all 4 markers are detected and the orientation is correct, check for stillness to trigger the capture and processing of the sheet.
            if isOrientationCorrect:
                if not orientationLocked:
                    if DEBUG:
                        print("1. Sheet is in correct orientation")
    
                    orientationLocked = True
                    prevGray = gray
                    stillFrames = 0

                if DEBUG:
                    cv2.putText(frame, "Correct Orientation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

###############################################################################################################
                                        # STILLNESS CHECK #
###############################################################################################################
                #Compare the difference between the previous frame and the new one.
                diff = cv2.absdiff(gray, prevGray)

                #Check that the paper is not moving.
                movement = np.sum(diff) / diff.size
                if DEBUG:
                    print("2. The movement is: ", movement)

                #When there's no movement, increment the still frame counter. If there's movement, reset it to 0.
                if movement < MOVEMENT_THRESHOLD:
                    if DEBUG:
                        print("3. stillFrame = ", stillFrames)
                    stillFrames += 1
                else:
                    stillFrames = 0

                #Set the current frame as the previous one
                prevGray = gray

                #Once we pass the stillness required, take the photo and analyze it. Reset other values
                if stillFrames > STILLNESS_REQUIRED and AUTO_CAPTURE:
                    stillFrames = 0

                    self.processor.SaveImage(PHOTO_DIR, "captured_image", frame)

###############################################################################################################
                                        # IMAGE PROCESSING + CNN Response #
###############################################################################################################
                    #1. Crop the image based on the centers found above
                    croppedImage = self.PerspectiveTransform(centers, frame)

                    if DEBUG:
                        self.processor.SaveImage(CROPPED_DIR, "cropped_image", croppedImage)

                    stats = {}
                    textFields = {}
                    for section in SKILL_SECTIONS:

                        #2. Crop the sheet further to separate the sections, to then perform image processing and digit segmentation on each section.
                        croppedSection = self.extractSection(croppedImage, section)

                        #3a. Get the Digit from the section
                        if section[0] == 0:
                            #Crop number area
                            roi = self.processor.ROI(croppedSection, section[1])
                            
                            #Perform the different Image Processing steps
                            processed = self.processor.ProcessImage(roi, section[1])    

                            #Perform image segmentation on the number
                            number = self.processor.SegmentDigit(processed, section[1])

                            #Detect numbers in each section using the trained CNN
                            numberStr = ""
                            for digitImg in number:
                                predicted = digitRecognizer.PredictDigit(digitImg)
                                numberStr += str(predicted)

                            stats[section[1]] = numberStr
                            if DEBUG:
                                print(f"{section[0]} value:", numberStr)

                        #3b. Get the characters from the section
                        elif section[0] == 1:
                            
                            #TEST FOR NOW
                            textFields[section[1]] = "Miguel the Wizard of Oz"
                            # if DEBUG:
                            #     print(f"{section[0]} value:", numberStr)

                    #4. Create a JSON file with the results
                    playerData = {
                        **textFields,
                        **stats
                    }

                    #JSON File
                    self.SavePlayerJSON(playerData)

                    #Get another player by returning true to the main
                    return True

            #When the orientation is not correct, reset the stillness and orientation flags.
            else:
                orientationLocked = False
                prevGray = None
                stillFrames = 0

                if DEBUG:
                    cv2.putText(frame, "Incorrect Orientation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Smart Document Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if DEBUG_CAMO:
            #Stop video capture.
            self.cap.release()
    
        else:
            #Stop PiCamera
            self.cap.stop()

        cv2.destroyAllWindows()

    #Create a JSON file based on the player's data
    def SavePlayerJSON(self, playerData, filePath=PLAYER_FILE):
        
        # Load existing data if file exists
        if os.path.exists(filePath):
            with open(filePath, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Determine next player number
        playerNumber = len(data) + 1
        playerKey = f"Player{playerNumber}"

        # Add new player
        data[playerKey] = playerData

        # Write updated JSON
        with open(filePath, "w") as f:
            json.dump(data, f, indent=2)

    #Deletes the JSON player file
    def DeletePlayerJSON(self, filePath=PLAYER_FILE):
        if filePath.exists():
            filePath.unlink()
            if DEBUG:
                print(f"Deleted JSON file: {filePath}")
        else:
            if DEBUG:
                print("JSON file does not exist.")

# print(isMorePlayers)
    # # Generate padded markers
    # ARUCO_MARKERS = 4
    # for marker_id in range(ARUCO_MARKERS):
    #     aruco_marker.GenerateArucoMarkers(marker_id)
