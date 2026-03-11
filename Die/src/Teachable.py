from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from .Constants import *

# Disable scientific notation
np.set_printoptions(suppress=True)

class Teachable:

    def __init__(self):
        # Load model once
        self.model = load_model(KERAS_H5, compile=False)

        # Load class labels
        with open(LABEL, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def TeachableMachine(self, img):
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        
        # Prepare input array
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.fromarray(img).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized

        # Predicts the model
        prediction = self.model.predict(data, verbose=0)

        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        return class_name[2:], confidence_score