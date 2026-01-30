from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "XRGB8888"})
picam2.configure(config)

picam2.start_preview(Preview.QTGL)
picam2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

picam2.stop()
