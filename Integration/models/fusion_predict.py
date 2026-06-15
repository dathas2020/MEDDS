import numpy as np
from tensorflow.keras.models import load_model

from config import (
    FUSION_MODEL_PATH,
    VOICE_WEIGHT,
    EMOTION_CLASSES,
    EMOTION_CALIBRATION
)

print("Loading Fusion Model...")

fusion_model = load_model(
    FUSION_MODEL_PATH,
    compile=False
)

print("Fusion Model Ready")


def predict_fusion(
    face_embedding,
    voice_embedding
):

    face_embedding = np.asarray(
        face_embedding,
        dtype=np.float32
    )

    voice_embedding = np.asarray(
        voice_embedding,
        dtype=np.float32
    )

    # alpha discovered during weight search
    voice_embedding = (
        voice_embedding *
        VOICE_WEIGHT
    )

    print(
        "Face norm:",
        np.linalg.norm(face_embedding)
    )

    print(
        "Voice norm:",
        np.linalg.norm(voice_embedding)
    )

    fused_embedding = np.concatenate(
        [
            face_embedding,
            voice_embedding
        ]
    )

    fused_embedding = np.expand_dims(
        fused_embedding,
        axis=0
    )

    probabilities = fusion_model.predict(
        fused_embedding,
        verbose=0
    )[0]

    probabilities = (
        probabilities *
        np.array(
            EMOTION_CALIBRATION,
            dtype=np.float32
        )
    )

    probabilities = (
        probabilities /
        np.sum(probabilities)
    )

    class_index = np.argmax(
        probabilities
    )

    emotion = EMOTION_CLASSES[
        class_index
    ]

    confidence = float(
        probabilities[class_index]
    ) * 100

    probability_dict = {}

    for emotion_name, prob in zip(
        EMOTION_CLASSES,
        probabilities
    ):
        probability_dict[
            emotion_name
        ] = round(
            float(prob) * 100,
            2
        )

    sorted_probs = sorted(
        probability_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTOP 3 EMOTIONS:")
    for emotion_name, score in sorted_probs[:3]:
        print(
            f"{emotion_name}: {score}%"
        )

    return {
        "emotion": emotion.title(),
        "confidence": round(confidence, 2),
        "probabilities": probability_dict,
        "top3": sorted_probs[:3]
    }