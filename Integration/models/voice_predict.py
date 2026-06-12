import pickle
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import traceback

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VOICE_MODEL_PATH = (
    PROJECT_ROOT /
    "Voice_Model" /
    "saved_models" /
    "voice_model_v3.keras"
)

VOICE_LABEL_PATH = (
    PROJECT_ROOT /
    "Voice_Model" /
    "saved_models" /
    "label_encoder.pkl"
)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("VOICE_MODEL_PATH:", VOICE_MODEL_PATH)
print("EXISTS:", VOICE_MODEL_PATH.exists())

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

# ------------------------------------------------------------------
# Constants from training
# ------------------------------------------------------------------

SAMPLE_RATE = 16000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

SILENCE_TOP_DB = 30

# ------------------------------------------------------------------
# Load once
# ------------------------------------------------------------------

print("Loading Voice Model...")
voice_model = tf.keras.models.load_model(
    VOICE_MODEL_PATH
)

print("Loading Label Encoder...")
with open(VOICE_LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("Loading YAMNet...")
yamnet_model = hub.KerasLayer(
    YAMNET_URL,
    trainable=False
)

print("Voice System Ready")

def load_audio(file_path):

    waveform, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return waveform.astype(np.float32)


def trim_silence(waveform):

    trimmed, _ = librosa.effects.trim(
        waveform,
        top_db=SILENCE_TOP_DB
    )

    if len(trimmed) < SAMPLE_RATE // 4:
        return waveform

    return trimmed


def normalize_length(waveform):

    n = len(waveform)

    if n == CLIP_SAMPLES:
        return waveform

    if n < CLIP_SAMPLES:

        pad_total = CLIP_SAMPLES - n

        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        return np.pad(
            waveform,
            (pad_left, pad_right),
            mode="constant"
        )

    start = (n - CLIP_SAMPLES) // 2

    return waveform[
        start:start + CLIP_SAMPLES
    ]


def preprocess_audio(file_path):

    waveform = load_audio(file_path)

    waveform = trim_silence(
        waveform
    )

    waveform = normalize_length(
        waveform
    )

    return waveform

def extract_embedding(waveform):

    scores, embeddings, spectrogram = (
        yamnet_model(waveform)
    )

    embedding = tf.reduce_mean(
        embeddings,
        axis=0
    )

    return embedding.numpy().astype(
        np.float32
    )

def predict_voice(audio_path):

    try:
        print("STEP 1: preprocess")

        waveform = preprocess_audio(audio_path)

        print("STEP 2: waveform shape")
        print(waveform.shape)

        embedding = extract_embedding(
            waveform
        )

        print("STEP 3: embedding")
        print(embedding.shape)

        embedding = np.expand_dims(
            embedding,
            axis=0
        )

        prediction = voice_model.predict(
            embedding,
            verbose=0
        )

        prediction = prediction[0]

        # reduce over-prediction of disgust and fear

        prediction[1] *= 0.5   # disgust
        prediction[2] *= 0.6   # fear

        prediction = prediction / np.sum(prediction)
        for idx in np.argsort(prediction)[::-1]:
            print(
                label_encoder.classes_[idx],
                round(float(prediction[idx] * 100), 2)
            )
        print("STEP 4: prediction")
        print(prediction)

        class_index = np.argmax(
            prediction
        )

        confidence = float(
            np.max(prediction)
        ) * 100

        emotion = (
            label_encoder.classes_
            [class_index]
        )

        print(
            "VOICE RESULT:",
            emotion,
            confidence
        )

        return {

            "emotion":
                emotion.title(),

            "confidence":
                round(confidence, 2),

            "success":
                True
        }

    

    except Exception as e:

        print("\n========== VOICE TRACEBACK ==========")
        traceback.print_exc()
        print("=====================================\n")

        return {
            "emotion": "Unknown",
            "confidence": 0,
            "success": False,
            "error": str(e)
        }