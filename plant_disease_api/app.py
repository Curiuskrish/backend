from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import uuid

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = "uploadimages"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")



# Load labels
with open("data/plant_disease.json", "r") as file:
    plant_disease = json.load(file)

# Preprocess the image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((160, 160))  # Match EfficientNet input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌿 Plant Disease Recognition API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_bytes = image.read()
    processed_image = preprocess_image(image_bytes)
    prediction = model.predict(processed_image)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    data = plant_disease[str(predicted_index)]

    return jsonify({
        "name": data["name"],
        "cause": data["cause"],
        "cure": data["cure"],
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
