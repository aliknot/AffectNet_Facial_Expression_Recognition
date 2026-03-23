import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input as resnet_preprocess
from PIL import Image
from mtcnn import MTCNN

# 1. Load the FER model
print("Loading FER model...")
try:
    model = keras.models.load_model("affectnet_fer_model.keras")
    print("FER Model loaded successfully!")
except Exception as e:
    print(f"Error loading FER model: {e}")
    model = None

# 2. Load the MTCNN Face Detector
print("Loading MTCNN Face Detector...")
try:
    face_detector = MTCNN()
    print("MTCNN Face Detector loaded successfully!")
except Exception as e:
    print(f"Error loading MTCNN Face Detector: {e}")
    face_detector = None

class_names = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


def predict_emotion(img_file):
    # Safety check
    if model is None or face_detector is None:
        raise RuntimeError("Models failed to load. Please check the server logs.")

    # Load and convert image to RGB
    img = Image.open(img_file).convert("RGB")

    # --- FACE DETECTION & CROPPING ---
    img_array_np = np.array(img)
    faces = face_detector.detect_faces(img_array_np)

    if faces:
        x, y, w, h = faces[0]["box"]
        x, y = max(0, x), max(0, y)
        img = img.crop((x, y, x + w, y + h))
    else:
        raise ValueError(
            "No face detected in the image. Please upload a clearer photo."
        )

    # Save a copy of the cropped PIL image to return to the UI
    cropped_face_image = img.copy()

    # --- PREPROCESSING ---
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet_preprocess(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Notice the third return value here!
    return predicted_class, float(confidence), cropped_face_image
