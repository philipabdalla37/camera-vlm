from pathlib import Path

# Define directories for saving images
BASE_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = BASE_DIR / "images" 


#Files
ORIGINAL_IMAGE = "die.jpg"
ROI_IMAGE = "roi_die.jpg"
PROCESSED_IMAGE = "processed_die.jpg"
KERAS_H5 = BASE_DIR / "models" / "keras_model.h5"
LABEL = BASE_DIR / "models" / "labels.txt"

#Debugging flag
DEBUG = True
DEBUG_CAMO = True

#Camo
DISPLAY_SIZE = (640, 360)
MOVEMENT_THRESHOLD = 6.0      # Adjust if needed
STILLNESS_REQUIRED = 10       # Frames of stillness before capture
CONTOUR_AREA_THRESHOLD = 1500
AUTO_CAPTURE = True