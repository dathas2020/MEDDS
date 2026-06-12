from pathlib import Path
from models.voice_predict import predict_voice

PROJECT_ROOT = Path(__file__).resolve().parent.parent

audio_file = (
    PROJECT_ROOT /
    "Voice_Model" /
    "last_recording.wav"
)

print(audio_file)

result = predict_voice(str(audio_file))

print(result)