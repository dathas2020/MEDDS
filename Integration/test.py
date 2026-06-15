import cv2

from models.face_embedding import (
    extract_face_embedding
)

from models.voice_embedding import (
    extract_voice_embedding
)

from models.fusion_predict import (
    predict_fusion
)

img = cv2.imread(
    r"D:\AI\MEDDS\Integration\test.jpg"
)

audio = (
    r"D:\AI\MEDDS\Integration\test.wav"
)

face_embedding = (
    extract_face_embedding(img)
)

voice_embedding = (
    extract_voice_embedding(audio)
)

result = predict_fusion(
    face_embedding,
    voice_embedding
)

print(result)