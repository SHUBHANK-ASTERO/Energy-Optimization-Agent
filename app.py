from flask import Flask, render_template, jsonify, request
from pathlib import Path
import pandas as pd
from model_utils import ensure_model, predict_usage, detect_anomalies, build_recommendations

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "energy_data.csv"

df = pd.read_csv(DATA)
model = ensure_model(df)

RATE_PER_KWH = 8.0
CO2_KG_PER_KWH = 0.71

@app.route("/")
def dashboard():
    data = df.copy()
    recent = data.tail(24)
    predicted = predict_usage(model, recent.iloc[-1])
    anomalies = detect_anomalies(data)
    tips = build_recommendations(predicted, recent.iloc[-1], anomalies)

    total_24 = float(recent["energy_kwh"].sum())
    cost = total_24 * RATE_PER_KWH
    co2 = total_24 * CO2_KG_PER_KWH
    savings = max(predicted - 3.0, 0) * 0.20

    return render_template(
        "index.html",
        total_24=round(total_24, 2),
        predicted=round(predicted, 2),
        cost=round(cost, 2),
        co2=round(co2, 2),
        savings=round(savings, 2),
        anomaly_count=len(anomalies),
        tips=tips,
        avg=round(float(data["energy_kwh"].mean()), 2),
        peak=round(float(data["energy_kwh"].max()), 2)
    )

@app.route("/api/usage")
def usage():
    recent = df.tail(24)
    return jsonify({
        "labels": recent["timestamp"].tolist(),
        "values": recent["energy_kwh"].round(2).tolist()
    })

@app.route("/api/devices")
def devices():
    latest = df.tail(24)
    cols = ["lighting_kwh", "ac_kwh", "computer_kwh", "other_kwh"]
    names = ["Lighting", "Air Conditioner", "Computers", "Other"]
    return jsonify({
        "labels": names,
        "values": [round(float(latest[c].sum()), 2) for c in cols]
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(silent=True) or {}
    row = pd.Series({
        "hour": float(body.get("hour", 14)),
        "temperature": float(body.get("temperature", 30)),
        "occupancy": float(body.get("occupancy", 10)),
        "previous_kwh": float(body.get("previous_kwh", 4))
    })
    prediction = predict_usage(model, row)
    return jsonify({
        "predicted_kwh": round(prediction, 2),
        "estimated_cost": round(prediction * RATE_PER_KWH, 2),
        "estimated_co2": round(prediction * CO2_KG_PER_KWH, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
