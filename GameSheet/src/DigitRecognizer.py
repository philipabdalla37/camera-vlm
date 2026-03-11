import tensorflow as tf
import numpy as np

class DigitRecognizer:

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def PredictDigit(self, digitImg):
        # Normalize to 0-1
        digitImg = digitImg.astype("float32") / 255.0

        # Add channel dimension
        digitImg = digitImg.reshape(1, 28, 28, 1)

        prediction = self.model.predict(digitImg, verbose=0)
        print("Confidence:", np.max(prediction))

        return np.argmax(prediction)