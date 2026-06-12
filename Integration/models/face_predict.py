import base64
import pickle

from io import BytesIO

from PIL import Image


# future imports
# import tensorflow as tf
# import numpy as np
import os


def predict_face(image_data):

    try:

        print("Current Directory:")
        print(os.getcwd())

        image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(
            image_data
        )

        image = Image.open(
            BytesIO(image_bytes)
        )

        os.makedirs(
            "uploads",
            exist_ok=True
        )

        from pathlib import Path

        PROJECT_ROOT = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        UPLOAD_DIR = (
            PROJECT_ROOT /
            "uploads"
        )

        UPLOAD_DIR.mkdir(
            exist_ok=True
        )

        frame_path = (
            UPLOAD_DIR /
            "latest_frame.jpg"
        )

        image.save(frame_path)

        # future:
        # image = preprocess(image)
        # prediction = model.predict(image)

        return {

            "emotion": "Happy",

            "confidence": 84,

            "success": True

        }

    except Exception as e:

        return {

            "emotion": "Unknown",

            "confidence": 0,

            "success": False,

            "error": str(e)

        }