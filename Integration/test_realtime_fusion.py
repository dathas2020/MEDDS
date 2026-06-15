import cv2

from models.face_embedding import extract_face_embedding
from models.voice_embedding import extract_voice_embedding
from models.fusion_predict import predict_fusion

IMAGE_PATH = r"D:\AI\MEDDS\Integration\test.jpg"
AUDIO_PATH = r"D:\AI\MEDDS\Integration\test.wav"

img = cv2.imread(IMAGE_PATH)

face_emb = extract_face_embedding(img)
voice_emb = extract_voice_embedding(AUDIO_PATH)

result = predict_fusion(face_emb, voice_emb)

print(result)