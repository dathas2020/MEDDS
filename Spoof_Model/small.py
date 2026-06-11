import pickle
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_PATH = os.path.join(
    CURRENT_DIR,
    "label_encoder.pkl"
)

with open(LABEL_PATH, "rb") as f:
    obj = pickle.load(f)

print(type(obj))
print(obj)