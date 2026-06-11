import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "spoof_model_v1.keras"
)

LABEL_ENCODER_PATH = os.path.join(
    CURRENT_DIR,
    "label_encoder.pkl"
)

print("Model Path:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

print("Label Encoder Path:", LABEL_ENCODER_PATH)
print("Exists:", os.path.exists(LABEL_ENCODER_PATH))

import cv2
import numpy as np
import tensorflow as tf
import pickle
import os

# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_SIZE = 224

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("Classes:", list(label_encoder.values()))

# ==========================================================
# FACE DETECTOR
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================================
# PREPROCESS FUNCTION
# ==========================================================

def preprocess_image(frame):
    """
    Preprocess image for MobileNetV2
    """

    # Convert BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert to float
    img = img.astype(np.float32)

    # MobileNetV2 preprocessing
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_spoof(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # Largest face
    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    face_crop = frame[
        y:y+h,
        x:x+w
    ]

    processed = preprocess_image(
        face_crop
    )

    prediction = model.predict(
        processed,
        verbose=0
    )[0]

    class_index = np.argmax(
        prediction
    )

    predicted_label = label_encoder[
        class_index
    ]

    confidence = float(
        prediction[class_index]
    )

    real_prob = float(
        prediction[0]
    )

    spoof_prob = float(
        prediction[1]
    )

    return (
        predicted_label,
        confidence,
        real_prob,
        spoof_prob,
        (x, y, w, h)
    )

# ==========================================================
# CAMERA
# ==========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

print("Camera object created")

if not cap.isOpened():
    print("Could not access webcam.")
    exit()

print("Camera opened successfully")

print("\n===================================")
print("SPOOF DETECTION STARTED")
print("SPACE -> Capture & Predict")
print("Q     -> Quit")
print("===================================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    display_frame = frame.copy()

    cv2.putText(
        display_frame,
        "SPACE = Predict | Q = Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Spoof Detection Camera",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ======================================================
    # SPACE KEY -> PREDICT
    # ======================================================

    if key == 32:

        # Save captured image
        cv2.imwrite(
            "captured_frame.jpg",
            frame
        )

        result = predict_spoof(frame)

        if result is None:

            print("\nNo Face Detected\n")

            continue

        (
            predicted_label,
            confidence,
            real_prob,
            spoof_prob,
            (x, y, w, h)
        ) = result

        print("\n===================================")
        print("PREDICTION RESULT")
        print("===================================")
        print(f"Prediction : {predicted_label}")
        print(f"Confidence : {confidence:.4f}")
        print(f"Real Prob  : {real_prob:.4f}")
        print(f"Spoof Prob : {spoof_prob:.4f}")
        print("===================================\n")

        result_frame = frame.copy()

        # Green for real
        if predicted_label.lower() == "real":
            color = (0, 255, 0)

        # Red for spoof
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            result_frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            result_frame,
            f"{predicted_label.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            result_frame,
            f"Confidence: {confidence:.2%}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            result_frame,
            f"Real: {real_prob:.2%}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_frame,
            f"Spoof: {spoof_prob:.2%}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "Prediction Result",
            result_frame
        )

    # ======================================================
    # Q KEY -> EXIT
    # ======================================================

    elif key == ord("q"):
        break

# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print("Program closed.")