import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Face_Model",
    "stage1_best_model.keras"
)

FACE_LABEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Face_Model",
    "label_encoder.pkl"
)

SPOOF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Spoof_Model",
    "spoof_model_v1.keras"
)

SPOOF_LABEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Spoof_Model",
    "label_encoder.pkl"
)

VOICE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Voice_Model",
    "saved_models",
    "voice_model_v3.keras"
)

VOICE_LABEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Voice_Model",
    "saved_models",
    "label_encoder.pkl"
)