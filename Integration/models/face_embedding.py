import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from config import FACE_IMAGE_SIZE

print("Loading EfficientNetB0...")

face_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(*FACE_IMAGE_SIZE, 3)
)

face_model.trainable = False

print("EfficientNetB0 Ready")


def preprocess_image(image):

    image = tf.convert_to_tensor(image)

    image = tf.image.resize(
        image,
        FACE_IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.keras.applications.efficientnet.preprocess_input(
        image
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


def extract_face_embedding(image):

    image = preprocess_image(image)

    embedding = face_model.predict(
        image,
        verbose=0
    )

    return embedding[0].astype(np.float32)