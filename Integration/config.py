from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =====================================================
# Existing Integration Models
# =====================================================

FACE_MODEL_PATH = (
    PROJECT_ROOT /
    "Face_Model" /
    "saved_models" /
    "face_model_v2.keras"
)

VOICE_MODEL_PATH = (
    PROJECT_ROOT /
    "Voice_Model" /
    "saved_models" /
    "voice_model_v3.keras"
)

SPOOF_MODEL_PATH = (
    PROJECT_ROOT /
    "Spoof_Model" /
    "saved_models" /
    "spoof_model.keras"
)

# =====================================================
# Fusion System
# =====================================================

FACE_IMAGE_SIZE = (224, 224)

WHISPER_MODEL_NAME = "openai/whisper-tiny"
WHISPER_SAMPLING_RATE = 16000

VOICE_WEIGHT = 1.5

FUSION_MODEL_PATH = (
    PROJECT_ROOT /
    "Emotion_Model" /
    "Fusion_Weight_Search_Output" /
    "final_fusion_model.keras"
)

EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

EMOTION_CALIBRATION = [
    0.01,   # angry
    0.01,   # disgust
    0.01,  # fear
    0.01,   # happy
    0.01,   # neutral
    0.01,   # sad
    0.01    # surprise
]

