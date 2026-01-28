import os
import shutil
from ImageProcessing import *
from PyTesseract import *
from Teachable import *
from GPT4o import *

#Die Image locations
src = "images/TableImages/6/6 t3.jpg"

#Different methods for die recognition
method = {
    "OCR": 1,
    "CNN": 2,
    "VLM": 3
}

# #Set current method 
# curMethod = method["VLM"]

# for image in os.listdir(src):
#     fullPath = os.path.join(src, image)

#     #PyTesseract selected
#     if curMethod == method["OCR"]:
#         output = runPyTesseract(fullPath)
        
#     #Teachable Machine selected
#     elif curMethod == method["CNN"]:
#         output = TeachableMachine(fullPath)
        
#     #GPT-4o selected
#     elif curMethod == method["VLM"]:
#         response = SendToGpt(fullPath)

response = SendToGpt(src)
# #Transform Normal Image to Processed Image
# for i in range(1,7):
#     j = 1
#     SRC = f'25cm/{i}/photos/'
#     DEST = f'25cm/{i}/NewProcessed/'

#     # Delete the old folder
#     if os.path.exists(DEST):
#         shutil.rmtree(DEST)   # removes the entire folder and all files inside

#     # Sort files numerically based on the number after the underscore
#     files = sorted(os.listdir(SRC), key=lambda x: int(x.split('_')[1].split('.')[0]))

#     for image in files:
    
#         fullPath = os.path.join(SRC, image)
#         print(fullPath)
#         ROI(fullPath)
#         ProcessImage()

#         #Folder where processed images will be copied
#         os.makedirs(DEST, exist_ok=True)

#         #Name of the processed image (assumed to be stored in PROCESSED_IMAGE)
#         if os.path.exists(PROCESSED_IMAGE):
#             new_name = f"processed{i}_{j}.jpg"
#             destination_path = os.path.join(DEST, new_name)
#             shutil.copy(PROCESSED_IMAGE, destination_path)
#             print(f"Copied processed image to {destination_path}")
#             j += 1


