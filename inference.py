import keras
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input as resnet_preprocess
import keras_hub
from PIL import Image

# 1. Load the complete model directly
print("Loading model...")
try:
    model = keras.models.load_model("affectnet_fer_model.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

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
    if model is None:
        raise RuntimeError("Model failed to load. Please check the model file path.")

    # Load and convert image to RGB
    img = Image.open(img_file).convert("RGB")

    # Resize to 224x224
    img = img.resize((224, 224))

    # Convert to array and add batch dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Apply ResNet normalization
    img_array = resnet_preprocess(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, float(confidence)
