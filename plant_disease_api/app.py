from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

# Setup upload folder
UPLOAD_FOLDER = "uploadimages"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the TensorFlow model
import gdown

# Model download path
model_path = "models/plant_disease_recog_model_pwp.keras"
model_drive_url = "https://drive.google.com/file/d/1qDqeP1rHcawATIR4sv3WRULHJUh-FUFO/view?usp=sharing"  # ⬅️ Replace with your actual file ID

# Ensure model is downloaded
if not os.path.exists(model_path):
    print("🔽 Downloading model from Google Drive...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gdown.download(model_drive_url, output=model_path, quiet=False)
    print("✅ Model downloaded successfully!")

# Load the model
model = tf.keras.models.load_model(model_path)


# Load label data
with open("data/plant_disease.json", "r") as file:
    plant_disease = json.load(file)

# Preprocess uploaded image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((160, 160))  # EfficientNet default input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Health check endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌿 Plant Disease Recognition API is running."})

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_bytes = image.read()
    processed_image = preprocess_image(image_bytes)

    # Predict using model
    prediction = model.predict(processed_image)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    data = plant_disease.get(str(predicted_index), {
        "name": "Unknown",
        "cause": "Unknown",
        "cure": "Unknown"
    })

    return jsonify({
        "name": data["name"],
        "cause": data["cause"],
        "cure": data["cure"],
        "confidence": round(confidence, 4)
    })



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if not set
    app.run(host="0.0.0.0", port=port)


