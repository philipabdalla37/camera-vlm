# ORIGINAL_IMAGE = "images/test/character_sheet4.png"
ORIGINAL_IMAGE = "images/test/test_sheet.png"
ROI_IMAGE = "images/test/test_sheet.png"
CROPPED_IMAGE = "images/croppedSheet/cropped_sheet.png"

#Camo
IMAGE_SIZE = (720, 960)
MOVEMENT_THRESHOLD = 2.0      # Adjust if needed
STILLNESS_REQUIRED = 10       # Frames of stillness before capture
ANGLE_TOLERANCE = 1.0        # Degrees of tolerance for orientation check
AUTO_CAPTURE = True

#ArUco IDs
ARUCO_MARKERS = 4
ARUCO_IDS = {0, 1, 2, 3}

#Sheet attributes [name, (x1, y1, x2, y2)]
SKILL_SECTIONS = [["strength", (20, 200, 150, 250)],
    ["intelligence", (20, 270, 150, 310)],
    ["charisma", (20, 350, 150, 390)],
    ["dexterity", (20, 433, 150, 473)]]

ROI_PADDING = (45, 20, 35, 13)  # (pad_x1, pad_y1, pad_x2, pad_y2)



