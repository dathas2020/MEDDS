from pydub import AudioSegment
from pydub.utils import which

latest_audio_path = None

latest_fusion_result = {
    "emotion": "Waiting",
    "confidence": 0,
    "probabilities": {}
}

AudioSegment.converter = (
    r"C:\ffmpeg-2026-06-08-git-6028720d70-essentials_build\bin\ffmpeg.exe"
)

import os

os.environ["PATH"] += os.pathsep + (
    r"C:\ffmpeg-2026-06-08-git-6028720d70-essentials_build\bin"
)

print("ffmpeg:", which("ffmpeg"))
print("ffprobe:", which("ffprobe"))

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_ROOT / "uploads"

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

from models.fusion_service import (
    predict_multimodal
)

latest_spoof_result = {
    "status": "Pending",
    "confidence": 0
}

latest_distraction_result = {
    "level": "Pending",
    "score": 0
}

app = Flask(__name__)

latest_face_detected = False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_predictions")
def get_predictions():

    return jsonify({

        "face_detected":
            latest_face_detected,

        "final_emotion":
            latest_fusion_result["emotion"],

        "final_confidence":
            latest_fusion_result["confidence"],

        "top3":
            latest_fusion_result.get(
                "top3",
                []
            ),

        "spoof_status":
            f'{latest_spoof_result["status"]} '
            f'({latest_spoof_result["confidence"]}%)',

        "distraction_level":
            latest_distraction_result["level"],

        "distraction_score":
            latest_distraction_result["score"]
    })

@app.route("/system_status")
def system_status():

    return jsonify({
        "face_model": True,
        "voice_model": True,
        "spoof_model": True,
        "models_loaded": 3,
        "total_models": 3
    })


@app.route("/predict_face", methods=["POST"])
def predict_face_api():

    data = request.get_json()

    result = predict_face(
        data["image"]
    )

    global latest_face_detected

    latest_face_detected = result.get(
        "face_detected",
        False
    )
    
    global latest_distraction_result

    latest_distraction_result = {

        "level":
            result.get(
                "distraction_level",
                "-"
            ),

        "score":
            result.get(
                "distraction_score",
                0
            )
    }

    if not latest_face_detected:

        global latest_fusion_result
        global latest_spoof_result
        

        latest_fusion_result = {
            "emotion": "No Face",
            "confidence": 0,
            "top3": []
        }

        latest_spoof_result = {
            "status": "-"
        }

        latest_distraction_result = {
            "level": "No Face Detected",
            "score": 100
        }

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
        
    print("WEBM PATH:", save_path)
    print("WAV PATH:", wav_path)

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

    frame_path = (
        UPLOAD_DIR /
        "latest_frame.jpg"
    )

    if frame_path.exists():

        from models.fusion_service import (
            predict_multimodal
        )

        global latest_fusion_result

        latest_fusion_result = (
            predict_multimodal(
                frame_path,
                wav_path
            )
        )
        
        global latest_spoof_result

        latest_spoof_result = {
            "status": "Pending",
            "confidence": 0
        }

        latest_spoof_result = predict_spoof(
            frame_path
        )

        print("\nLATEST FUSION RESULT:")
        print(latest_fusion_result)

        print(
            "FUSION RESULT:",
            latest_fusion_result
        )

    result = predict_voice(
        wav_path
    )

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)