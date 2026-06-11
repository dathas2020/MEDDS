import pickle

with open(
    r"D:\AI\MEDDS\Voice_Model\saved_models\label_encoder.pkl",
    "rb"
) as f:
    le = pickle.load(f)

print(type(le))

try:
    print(le.classes_)
except:
    print(le)