from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = (
    r"C:\ffmpeg-2026-06-08-git-6028720d70-essentials_build\bin\ffmpeg.exe"
)

import os

os.environ["PATH"] += os.pathsep + (
    r"C:\ffmpeg-2026-06-08-git-6028720d70-essentials_build\bin"
)

print("ffmpeg:", which("ffmpeg"))
print("ffprobe:", which("ffprobe"))

import os
from werkzeug.utils import secure_filename

from config import (
    FACE_MODEL_PATH,
    VOICE_MODEL_PATH,
    SPOOF_MODEL_PATH
)

from flask import request
from flask import Flask, render_template, jsonify

from models.face_predict import predict_face
from models.voice_predict import predict_voice
from models.spoof_predict import predict_spoof

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_predictions")
def get_predictions():

    return jsonify({

        "face_emotion": "Waiting",
        "face_confidence": 0,

        "voice_emotion": "Waiting",
        "voice_confidence": 0,

        "spoof_status": "Waiting",
        "spoof_confidence": 0,

        "final_emotion": "Waiting"

    })

@app.route("/system_status")
def system_status():

    print("FACE:", FACE_MODEL_PATH)
    print("VOICE:", VOICE_MODEL_PATH)
    print("SPOOF:", SPOOF_MODEL_PATH)

    face_exists = os.path.exists(FACE_MODEL_PATH)
    voice_exists = os.path.exists(VOICE_MODEL_PATH)
    spoof_exists = os.path.exists(SPOOF_MODEL_PATH)

    print(face_exists, voice_exists, spoof_exists)

    return jsonify({
        "face_model": face_exists,
        "voice_model": voice_exists,
        "spoof_model": spoof_exists,
        "models_loaded": sum([
            face_exists,
            voice_exists,
            spoof_exists
        ]),
        "total_models": 3
    })


@app.route("/predict_face", methods=["POST"])
def predict_face_api():

    data = request.get_json()

    result = predict_face(
        data["image"]
    )

    return jsonify(result)

from pydub import AudioSegment

@app.route("/predict_voice", methods=["POST"])
def predict_voice_api():

    upload_dir = os.path.join(
        "uploads",
        "audio"
    )

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    save_path = os.path.join(
        upload_dir,
        "latest_audio.webm"
    )

    audio_file = request.files["audio"]

    audio_file.save(save_path)

    wav_path = os.path.join(
        upload_dir,
        "latest_audio.wav"
    )

    audio = AudioSegment.from_file(
        save_path,
        format="webm"
    )

    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)

    audio.export(
        wav_path,
        format="wav"
    )

    result = predict_voice(
        wav_path
    )

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)