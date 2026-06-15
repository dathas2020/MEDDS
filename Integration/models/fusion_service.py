import cv2
import numpy as np

from models.face_embedding import extract_face_embedding
from models.voice_embedding import extract_voice_embedding
from models.fusion_predict import predict_fusion


def predict_multimodal(
    image_path,
    audio_path
):

    image = cv2.imread(
        str(image_path)
    )

    face_embedding = extract_face_embedding(image)

    #voice_embedding = (
    #    extract_voice_embedding(
    #        str(audio_path)
    #    )
    #)
    voice_embedding = extract_voice_embedding(
        str(audio_path)
    )
    print(
        "VOICE EMBEDDING NORM:",
        np.linalg.norm(voice_embedding)
    )

    result = predict_fusion(
        face_embedding,
        voice_embedding
    )

    return result