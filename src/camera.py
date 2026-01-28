import cv2
import os
import shutil
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from ImageProcessing import *
from Teachable import *
from GPT4o import *

#Different methods for die recognition
method = {
    "OCR": 1,
    "CNN": 2,
    "VLM": 3
}

#Set current method 
curMethod = method["CNN"]

#global variables
isPaused = False

curFrame = None
i = 1

# Initialize OpenCV capture
cap = cv2.VideoCapture(0)           #1 to detect Camo Camera for Iphone

# Tkinter window
root = tk.Tk()
root.title("Camera Capture")

# Label to show the camera frame
videoLabel = ttk.Label(root)
videoLabel.pack()

# Function to show camera frame
def showFrame():
    global isPaused, curFrame
    ret, frame = cap.read()
    if ret and isPaused == False:

        #Get the most recent frame
        curFrame = frame.copy()
        
        # Convert frame to RGB for Tkinter
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        #Update the label with this image
        videoLabel.imgtk = imgtk
        videoLabel.configure(image=imgtk)

        #Wait 10ms, then call this function again so that the frames keep updating
        videoLabel.after(10, showFrame)

# Function to resume camera feed
def onResume():
    global isPaused
    resumeButton.pack_forget()  # Hides the resume button
    isPaused = False
    showFrame()                 # Resumes the camera feed

# Function to capture and save frame
def onCaptureFrame():
    global isPaused, curFrame
    if curFrame is not None:
        resumeButton.pack(pady=10)  # Shows the resume button
        cv2.imwrite(ORIGINAL_IMAGE, curFrame)
        print("Image saved as snapshot.jpg")
        isPaused = True
        runImageProcessing(ORIGINAL_IMAGE)

# Function to process the captured image
def runImageProcessing(image):
    ROI(image)
    ProcessImage(ROI_IMAGE)

    #Teachable Machine selected
    if curMethod == method["CNN"]:
        output = TeachableMachine(PROCESSED_IMAGE)
        messagebox.showinfo("Result", output)

#Adds a Button that takes screenshot
screenshotButton = tk.Button(root, text="Capture Frame", command=onCaptureFrame)
screenshotButton.pack(pady=10)

resumeButton = tk.Button(root, text="Resume Camera", command=onResume)

# Start the video loop
showFrame()

# Runs app
root.mainloop()

# Cleanup after window is closed
cap.release()
cv2.destroyAllWindows()


# # USE WHEN MORE PROCESSED IMAGES NEEDED ##
# # This speeds up the process of taking the images ##
# def runImageProcessing(image):
#     global i
#     outputFolder = "25cm/5/photos/"
#     outputProcessedFolder = "25cm/5/NewProcessed/"

#     #Folder where processed images will be copied
#     os.makedirs(outputFolder, exist_ok=True)

#     #Name of the processed image (assumed to be stored in PROCESSED_IMAGE)
#     if os.path.exists(image):
#         new_name = f"die5_{i}.jpg"
#         destination_path = os.path.join(outputFolder, new_name)
#         shutil.copy(image, destination_path)
#         print(f"Copied processed image to {destination_path}")

#     ROI(image)
#     ProcessImage(ROI_IMAGE)

#     #PyTesseract selected
#     if curMethod == method["OCR"]:
#         output = runPyTesseract(PROCESSED_IMAGE)
#         messagebox.showinfo("Result", output)
    
#     #Teachable Machine selected
#     elif curMethod == method["CNN"]:
#         output = TeachableMachine(PROCESSED_IMAGE)
#         messagebox.showinfo("Result", output)

#     #GPT-4o selected
#     elif curMethod == method["VLM"]:
#         response = SendToGpt(PROCESSED_IMAGE)
#         messagebox.showinfo("GPT-4o Response", response)


#     #Folder where processed images will be copied
#     os.makedirs(outputProcessedFolder, exist_ok=True)

#     #Name of the processed image (assumed to be stored in PROCESSED_IMAGE)
#     if os.path.exists(PROCESSED_IMAGE):
#         new_name = f"processed5_{i}.jpg"
#         destination_path = os.path.join(outputProcessedFolder, new_name)
#         shutil.copy(PROCESSED_IMAGE, destination_path)
#         print(f"Copied processed image to {destination_path}")
#         i += 1