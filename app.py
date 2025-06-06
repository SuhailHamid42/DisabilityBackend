from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

# Load model and helpers
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return "ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        features = np.array(data["features"]).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
