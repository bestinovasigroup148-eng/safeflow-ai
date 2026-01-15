# app.py
# -------------------------------------------------
# SafeFlow AI Server (Cloud Deployment Version)
# Receives waterLevelMM and rateMM from ESP32/Wokwi
# Returns validated AI predictions for flood forecasting
# -------------------------------------------------

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# ------------------ LOAD MODEL ------------------
MODEL_PATH = "flood_model.h5"

# Load model for inference only (cloud-safe)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ SafeFlow AI model loaded")

# ------------------ FLASK APP ------------------
app = Flask(__name__)

# ------------------ PREDICTION LOGIC ------------------
def predict_flood(waterLevelMM, rateMM):
    """
    waterLevelMM : float (can be rising or falling)
    rateMM       : float (negative allowed)
    """

    # Prepare input
    X = np.array([[waterLevelMM, rateMM]], dtype=np.float32)
    y = model.predict(X, verbose=0)[0]

    # ---------- REGRESSION OUTPUTS ----------
    acceleration = float(y[0])
    time_above_threshold = max(0.0, float(y[1]))
    predicted_water_level = float(y[2])
    confidence = float(np.clip(y[3], 0.0, 1.0))
    time_to_danger = max(0.0, float(y[4]))

    # ---------- CLASSIFICATION OUTPUTS ----------
    trend_type = int(np.clip(round(y[5]), 0, 2))        # 0=stable,1=rising,2=falling
    anomaly_flag = int(np.clip(round(y[6]), 0, 1))      # 0=no anomaly,1=anomaly
    seasonal_pattern = int(np.clip(round(y[7]), 0, 23)) # hour of day
    risk_category = int(np.clip(round(y[8]), 0, 2))     # 0=safe,1=alert,2=danger

    return {
        "acceleration": acceleration,
        "timeAboveThreshold": time_above_threshold,
        "predictedWaterLevel": predicted_water_level,
        "confidence": confidence,
        "timeToDanger": time_to_danger,
        "trendType": trend_type,
        "anomalyFlag": anomaly_flag,
        "seasonalPattern": seasonal_pattern,
        "riskCategory": risk_category
    }

# ------------------ ROUTES ------------------
@app.route("/", methods=["GET"])
def index():
    return "SafeFlow AI Server is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    if not data or "waterLevelMM" not in data or "rateMM" not in data:
        return jsonify({"error": "Missing waterLevelMM or rateMM"}), 400

    try:
        waterLevelMM = float(data["waterLevelMM"])
        rateMM = float(data["rateMM"])
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid numeric input"}), 400

    result = predict_flood(waterLevelMM, rateMM)
    return jsonify(result)

# -------------------------------------------------
# IMPORTANT:
# DO NOT use app.run() here.
# Gunicorn (Render/Railway) will run the app.
# -------------------------------------------------
