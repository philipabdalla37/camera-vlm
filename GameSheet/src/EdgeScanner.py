from email.mime import image

import cv2
import numpy as np
from Constants import *
from ImageProcessing import *

class ArucoMarkers:
    
    # Generates a padded ArUco marker for the given ID and saves it as a PNG file.
    def GenerateArucoMarkers(self, marker_id, aruco_type=cv2.aruco.DICT_6X6_250, marker_size=800, border_bits=1):
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, borderBits=border_bits)
        
        # Add external white padding
        margin = int(marker_size * 0.25)
        padded_size = marker_size + 2 * margin
        padded = 255 * np.ones((padded_size, padded_size), dtype=np.uint8)
        padded[margin:margin + marker_size, margin:margin + marker_size] = marker
        filename = f"aruco_marker_{marker_id}_padded.png"
        cv2.imwrite(filename, padded)

    # Detects ArUco markers in the frame and returns their centers and detection data.
    def DetectMarkers(self, aruco_type=cv2.aruco.DICT_6X6_250):

        # Each marker contains a 6×6 binary grid, which allows for a total of 250 unique markers in the DICT_6X6_250 dictionary.
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)

        # Creates a parameter object controlling detection behavior.
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # Combines the dictionary and parameters to create a marker detector object. This object will be used to detect markers in the video feed.
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        cap = cv2.VideoCapture(0)

        # Force higher resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    

        # Flags and variables to check orientation and stillness
        orientation_locked = False
        prevGray = None
        stillFrames = 0

        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Detect markers in the current frame. The detectMarkers method returns the corners 
            # of detected markers, and their corresponding IDs.
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)

            centers = {}
            
            ##################
            # DETECT MARKERS #
            ##################
            if marker_ids is not None:
                #Convert  to 1D array for easier processing
                marker_ids = marker_ids.flatten()
                # print("Detected IDs:", marker_ids)

                #Compute the center of each marker. zip() method allows to 
                #associate each set of corners with its corresponding marker ID.
                for corners, marker_id in zip(marker_corners, marker_ids):
                    #Get the first point set of the given marker 
                    pts = corners[0]
                    center = np.mean(pts, axis=0)
                    centers[marker_id] = center

                #Draws a green bounding box around each detected marker and labels it with its ID.
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

            #####################
            # ORIENTATION CHECK #
            #####################
            isOrientationCorrect = False

            #Check that all 4 markers are detected
            if set(centers.keys()) == ARUCO_IDS:
                c0 = centers[0]
                c1 = centers[1]
                c2 = centers[2]

                #Get the top and left vector
                v_top = c1 - c0
                v_left = c2 - c0

                #Get the angles of the top and left vectors
                angle_top = np.degrees(np.arctan2(v_top[1], v_top[0]))
                angle_left = np.degrees(np.arctan2(v_left[1], v_left[0]))

                print("Angles: top =", angle_top, ", left =", angle_left)
                cond_top = abs(angle_top) < ANGLE_TOLERANCE
                cond_left = abs(abs(angle_left) - 90) < ANGLE_TOLERANCE
            
                #Correct orientation if all 4 conditions are true.
                if cond_top and cond_left:
                    isOrientationCorrect = True

            #####################
            # STATE TRANSITIONS #
            #####################

            #Results
            if isOrientationCorrect:
                if not orientation_locked:
                    print("1. Sheet is in correct orientation")
                    orientation_locked = True
                    prevGray = gray
                    stillFrames = 0

                cv2.putText(frame, "Correct Orientation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ###################
            # STILLNESS CHECK #
            ###################
                #Compare the difference between the previous frame and the new one.
                diff = cv2.absdiff(gray, prevGray)

                print("2. The difference between the last two frames is: ", diff)
                    
                #Check that the paper is not moving.
                movement = np.sum(diff) / diff.size
                print("3. The movement is: ", movement)

                if movement < MOVEMENT_THRESHOLD:
                    print("4. Movement is less than threshold, thus add stillFrame = ", stillFrames)
                    stillFrames += 1
                else:
                    stillFrames = 0

                #Set the current frame as the previous one
                prevGray = gray

                #Once we pass the stillness required, take the photo and analyze it. Reset other values
                if stillFrames > STILLNESS_REQUIRED and AUTO_CAPTURE:
                    stillFrames = 0

                    print("Capturing image...")
                    cv2.imwrite("captured_image.jpg", frame)

                    #1. Crop the image based on the three dots found above
                    croppedImage = self.CropSheet(centers)
                    cv2.imwrite(CROPPED_IMAGE, croppedImage)

                    reference = centers[0]               
                    x0 = reference[0]
                    y0 = reference[1]

                    #Sheet attributes [name, (x1, y1, x2, y2)]
                    skills = [["strength", (x0 + x0*2, y0 + y0*2, x0 + x0*2, y0 + y0*2)],
                    ["intelligence", (x0 + x0*0.1, y0 + 270, x0 + x0*0.2, y0 + 310)],
                    ["charisma", (x0 + x0*0.1, y0 + 350, x0 + x0*0.2, y0 + 390)],
                    ["dexterity", (x0 + x0*0.1, y0 + 433, x0 + x0*0.2, y0 + 473)]]

                    #2. Crop the sheet further to separate the sections. Perform the Image Processing steps on each section and save them as PNG
                    for section in skills:
                        #extract section
                        extractSection(croppedImage, section)
                        
                        #Crop number area
                        ROI(f"images/sections/{section[0]}.png", section[0])
                        
                        #Perform Image Processing steps
                        ProcessImage(f"images/roi/{section[0]}.png")    

                        #3. Detect numbers in each section using CNN
                        # output = detectNumber(f"images/processedSections/{section[0]}.png")
                        # messagebox.showinfo(f"Result for {section[0]}", output)
        
                        #4q. Create a JSON file with the results
            else:
                orientation_locked = False
                prevGray = None
                stillFrames = 0
                cv2.putText(frame, "Incorrect Orientation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Smart Document Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def CropSheet(self, centers):
        # Load the image
        image = cv2.imread("captured_image.jpg")

        # Convert to integers
        x0 = int(centers[0][0])
        y0 = int(centers[0][1])
        x3 = int(centers[3][0])
        y3 = int(centers[3][1])

        croppedImage = image[y0:y3, x0:x3]
        return croppedImage
    
def extractSection(img, section):

    #Get section name and coordinates
    name = section[0]
    x1, y1, x2, y2 = section[1]

    #Crop the image based on section
    sectionImage = img[y1:y2, x1:x2]
    
    # Save the section as a PNG file
    cv2.imwrite(f'images/sections/{name}.png', sectionImage)
    cv2.waitKey(0)
    return sectionImage


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    aruco_marker = ArucoMarkers()
    ARUCO_MARKERS = 4

    # Generate padded markers
    for marker_id in range(ARUCO_MARKERS):
        aruco_marker.GenerateArucoMarkers(marker_id)

    aruco_marker.DetectMarkers()


