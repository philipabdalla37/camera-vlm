# ORIGINAL_IMAGE = "images/test/character_sheet4.png"
# ORIGINAL_IMAGE = "images/test/test_sheet.png"
# ROI_IMAGE = "images/test/test_sheet.png"
# CROPPED_IMAGE = "camera-vlm/GameSheet/images/croppedSheet/cropped_sheet.png"
from pathlib import Path

# Define directories for saving images
BASE_DIR = Path(__file__).resolve().parent.parent

#CNN Models
DIGIT_H5 = "camera-vlm/GameSheet/models/digit_model.h5"
# CHAR_H5 = BASE_DIR / "models" / "char_model.h5"

#Folders
IMAGES_DIR = BASE_DIR / "images"                        # Image folder    
PHOTO_DIR = IMAGES_DIR / "photo"                        # Sheet photo folder
CROPPED_DIR = IMAGES_DIR / "croppedSheet"               # Cropped sheet folder
SECTIONS_DIR = IMAGES_DIR / "sections"                  # Cropped images folder
ROI_DIR = IMAGES_DIR / "roi"                            # ROI image folder
PROCESSED_DIR = IMAGES_DIR / "processedSections"        # Processed image folder
MARKERS_DIR = IMAGES_DIR / "markers"                    # Markers folder
DEBUG_DIR = IMAGES_DIR / "debug"                        # Markers folder

#Debugging flag
DEBUG = True
DEBUG_CAMO = True

#Camo
MOVEMENT_THRESHOLD = 2.0      # Adjust if needed
STILLNESS_REQUIRED = 10       # Frames of stillness before capture
ANGLE_TOLERANCE = 1.0        # Degrees of tolerance for orientation check
AUTO_CAPTURE = True

#ArUco IDs
ARUCO_MARKERS = 4
ARUCO_IDS = {0, 1, 2, 3}

#Sheet attributes [0/1, name, (x1, y1, x2, y2)]
#0 -> Use Digit Segmentation
# 1 -> Use Character Segmentation
SKILL_SECTIONS = [[0, "strength", (16, 185, 150, 235)],
    [0, "intelligence", (16, 240, 150, 290)],
    [0, "charisma", (16, 300, 150, 350)],
    [0, "dexterity", (16, 360, 150, 410)],
    [1, "name", (232, 100, 532, 132)]]

ROI_PADDING = (45, 17, 50, 13)  # (pad_x1, pad_y1, pad_x2, pad_y2)



