import numpy as np
import cv2
import base64
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from ImageProcessing import *
from Teachable import *
from PyTesseract import *
import shutil
import time

# Globals
isPaused = False
curFrame = None
i = 1

dieDetected = False
prevGray = None
stillFrames = 0
AUTO_CAPTURE = True
MOVEMENT_THRESHOLD = 2.0      # Adjust if needed
STILLNESS_REQUIRED = 10       # Frames of stillness before capture

cap = cv2.VideoCapture(0)

# Tkinter window
root = tk.Tk()
root.title("Camera Capture")

videoLabel = ttk.Label(root)
videoLabel.pack()


# --------------------------------------------------------
# 1. Detect if a die is present using your ROI logic
# --------------------------------------------------------
def dieInFrame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours) > 0


# --------------------------------------------------------
# 2. Show frame + detect die + detect stillness + auto-capture
# --------------------------------------------------------
def showFrame():
    global isPaused, curFrame, prevGray, dieDetected, stillFrames

    ret, frame = cap.read()
    if ret and not isPaused:

        curFrame = frame.copy()

        # Detect die in frame
        if dieInFrame(frame):
            if not dieDetected:
                print("Die detected — waiting for it to stop rolling...")
                dieDetected = True
                prevGray = None
                stillFrames = 0

        else:
            dieDetected = False

        # If die is present, check motion to see when it stops
        if dieDetected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prevGray is not None:
                diff = cv2.absdiff(gray, prevGray)
                movement = np.sum(diff) / diff.size

                # Detect low movement = die is still
                if movement < MOVEMENT_THRESHOLD:
                    stillFrames += 1
                else:
                    stillFrames = 0

                # If die is still long enough → auto-capture
                if stillFrames > STILLNESS_REQUIRED and AUTO_CAPTURE:
                    print("Die stopped — capturing automatically.")
                    onCaptureFrame()
                    dieDetected = False
                    stillFrames = 0

            prevGray = gray

        # Display on GUI
        display = cv2.resize(frame, (640, 480))
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        videoLabel.imgtk = imgtk
        videoLabel.configure(image=imgtk)

        videoLabel.after(10, showFrame)


# --------------------------------------------------------
# 3. Capture and process frame
# --------------------------------------------------------
def onCaptureFrame():
    global isPaused, curFrame

    if curFrame is None or isPaused:
        return

    isPaused = True
    resumeButton.pack(pady=10)

    cv2.imwrite("snapshot.jpg", curFrame)
    print("Image saved as snapshot.jpg")

    runImageProcessing("snapshot.jpg")


def onResume():
    global isPaused
    resumeButton.pack_forget()
    isPaused = False
    showFrame()


def runImageProcessing(image):
    ROI(image)
    ProcessImage()
    output = runPyTesseract()
    TeachableMachine()
    messagebox.showinfo("Result", output)


# --------------------------------------------------------
# GUI Buttons
# --------------------------------------------------------
screenshotButton = tk.Button(root, text="Capture Frame", command=onCaptureFrame)
screenshotButton.pack(pady=10)

resumeButton = tk.Button(root, text="Resume Camera", command=onResume)


# Start video loop
showFrame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
