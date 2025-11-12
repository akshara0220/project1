from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keras.models import load_model
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__, template_folder='templates')
CORS(app)

# Load model and labels
model = load_model("food_model.h5")

if os.path.exists("class_indices.json"):
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    class_names = [name for _, name in sorted((v, k) for k, v in class_indices.items())]
else:
    class_names = ["Unknown"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(request.files['image']).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        predictions = model.predict(img_array)[0]
        idx = np.argmax(predictions)
        confidence = round(float(predictions[idx]) * 100, 2)
        label = class_names[idx]

        return jsonify({
            "food": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
