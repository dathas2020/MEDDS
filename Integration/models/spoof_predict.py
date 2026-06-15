import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

MODEL_PATH = (
    PROJECT_ROOT /
    "Spoof_Model" /
    "spoof_model_v1.keras"
)

print("Loading Spoof Model...")

spoof_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Spoof Model Ready")


def predict_spoof(image_path):

    try:

        image = cv2.imread(
            str(image_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (224, 224)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        prediction = spoof_model.predict(
            image,
            verbose=0
        )[0]

        print(
            "SPOOF RAW:",
            prediction
        )

        class_index = np.argmax(
            prediction
        )

        confidence = float(
            prediction[class_index]
        ) * 100

        if class_index == 0:

            status = "Real"

        else:

            status = "Spoof"

        return {

            "status":
                status,

            "confidence":
                round(confidence, 2)
        }

    except Exception as e:

        print(
            "Spoof Error:",
            e
        )

        return {

            "status":
                "Error",

            "confidence":
                0
        }