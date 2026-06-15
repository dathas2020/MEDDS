import base64
import pickle

from io import BytesIO

from PIL import Image


# future imports
# import tensorflow as tf
# import numpy as np
import os
import cv2
import numpy as np


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

        image_np = np.array(image)

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4
        )

        print("Faces found:", len(faces))

        if len(faces) == 0:

            return {

                "success": False,

                "face_detected": False

            }
        
        x, y, w, h = faces[0]

        img_h, img_w = image_np.shape[:2]

        face_center_x = x + (w / 2)
        face_center_y = y + (h / 2)

        image_center_x = img_w / 2
        image_center_y = img_h / 2

        dx = abs(
            face_center_x -
            image_center_x
        )

        dy = abs(
            face_center_y -
            image_center_y
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

        if dx < img_w * 0.15 and dy < img_h * 0.15:

            distraction_level = "Focused"
            distraction_score = 10

        elif dx < img_w * 0.25 and dy < img_h * 0.25:

            distraction_level = "Slightly Distracted"
            distraction_score = 40

        else:

            distraction_level = "Distracted"
            distraction_score = 80

        return {

            "success": True,

            "face_detected": True,

            "distraction_level":
                distraction_level,

            "distraction_score":
                distraction_score
        }

    except Exception as e:

        return {

            "emotion": "Unknown",

            "confidence": 0,

            "success": False,

            "error": str(e)

        }