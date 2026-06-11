import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd

from scipy.io.wavfile import write

# ==========================================================
# PATHS
# ==========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "saved_models",
    "voice_model_v3.keras"
)

LABEL_PATH = os.path.join(
    CURRENT_DIR,
    "saved_models",
    "label_encoder.pkl"
)

# ==========================================================
# CONFIG
# ==========================================================

RECORD_SECONDS = 3
MIC_SAMPLE_RATE = 48000

# Your microphone device
sd.default.device = (14, None)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading classifier model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Classifier loaded!")

# ==========================================================
# LOAD YAMNET
# ==========================================================

print("Loading YAMNet...")

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print("YAMNet loaded!")

# ==========================================================
# LOAD LABELS
# ==========================================================

with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_

print("Classes:", class_names)

# ==========================================================
# RECORD AUDIO
# ==========================================================

def record_audio():

    print("\nSpeak now...")

    recording = sd.rec(
        int(RECORD_SECONDS * MIC_SAMPLE_RATE),
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    audio = recording.flatten()

    print("Recording complete!")

    print(
        "Max amplitude:",
        np.max(np.abs(audio))
    )

    wav_path = os.path.join(
        CURRENT_DIR,
        "last_recording.wav"
    )

    write(
        wav_path,
        MIC_SAMPLE_RATE,
        (audio * 32767).astype(np.int16)
    )

    return wav_path

# ==========================================================
# YAMNET EMBEDDING
# ==========================================================

def extract_embedding(audio_path):

    waveform, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    waveform = waveform.astype(
        np.float32
    )

    rms = np.sqrt(
        np.mean(waveform**2)
    )

    print("RMS Energy:", rms)

    if rms < 0.01:
        return None

    scores, embeddings, spectrogram = yamnet_model(
        waveform
    )

    embeddings = embeddings.numpy()

    embedding = np.mean(
        embeddings,
        axis=0
    )

    embedding = np.expand_dims(
        embedding,
        axis=0
    )

    print(
        "Embedding Shape:",
        embedding.shape
    )

    return embedding

# ==========================================================
# PREDICT
# ==========================================================

def predict_emotion(audio_path):

    embedding = extract_embedding(
        audio_path
    )

    if embedding is None:
        return "No Speech Detected", None

    prediction = model.predict(
        embedding,
        verbose=0
    )[0]

    print("\nRaw Prediction:")

    top3 = np.argsort(prediction)[-3:][::-1]

    print("\nTop 3 Predictions:")

    for idx in top3:
        print(
            f"{class_names[idx]}: "
            f"{prediction[idx]*100:.2f}%"
        )

    for i, p in enumerate(prediction):

        print(
            f"{class_names[i]:<10}: {p:.4f}"
        )

    best_idx = np.argmax(
        prediction
    )

    confidence = prediction[
        best_idx
    ]

    # Confidence threshold
    if confidence < 0.50:

        predicted_label = "neutral"

    else:

        predicted_label = class_names[
            best_idx
        ]

    return (
        predicted_label,
        prediction
    )

# ==========================================================
# MAIN LOOP
# ==========================================================

with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print(label_encoder.classes_)

while True:

    input(
        "\nPress ENTER to record..."
    )

    audio_file = record_audio()

    emotion, probs = predict_emotion(
        audio_file
    )

    if probs is None:

        print(
            "\nNo Speech Detected"
        )

        continue

    print("\n================================")
    print("VOICE EMOTION RESULT")
    print("================================")

    for idx, prob in enumerate(probs):

        print(
            f"{class_names[idx]:<10}: {prob*100:.2f}%"
        )

    print(
        f"\nDominant Emotion: {emotion}"
    )

    print("================================")

    choice = input(
        "\nRecord Again? (y/n): "
    )

    if choice.lower() != "y":
        break