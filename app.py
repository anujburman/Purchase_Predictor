from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request body"}), 400

    sequence = np.array(data["sequence"])

    if sequence.shape != (7, 5):
        return jsonify({"error": f"Expected shape (7, 5), got {sequence.shape}"}), 400

    sequence_scaled = scaler.transform(sequence)
    sequence_scaled = sequence_scaled.reshape(1, 7, 5)

    prob = float(model.predict(sequence_scaled, verbose=0)[0][0])
    label = "Yes" if prob >= 0.5 else "No"

    return jsonify({
        "will_buy_probability": round(prob, 4),
        "prediction": label,
        "confidence": f"{round(prob * 100, 1)}%"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)