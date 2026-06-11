import tensorflow as tf
import tensorflow_hub as hub

print("TensorFlow:", tf.__version__)

print("Loading YAMNet...")

yamnet = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print("YAMNet Loaded Successfully!")