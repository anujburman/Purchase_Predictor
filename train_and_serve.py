import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify
import os

MODEL_PATH  = "lstm_model_server.keras"
SCALER_PATH = "scaler_server.pkl"

def train():
    print("Generating data...")
    np.random.seed(42)
    n_customers, n_days = 1000, 30
    records = []
    for cust_id in range(n_customers):
        for day in range(n_days):
            records.append({
                "customer_id": cust_id, "day": day,
                "pages_viewed":     np.random.poisson(3),
                "time_spent_min":   round(np.random.exponential(5), 2),
                "items_in_cart":    np.random.randint(0, 6),
                "discount_clicked": np.random.binomial(1, 0.3),
                "returned_visit":   np.random.binomial(1, 0.4),
            })
    df = pd.DataFrame(records)
    df = df.sort_values(["customer_id", "day"]).reset_index(drop=True)
    df["cart_trend"] = df.groupby("customer_id")["items_in_cart"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    df["time_trend"] = df.groupby("customer_id")["time_spent_min"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    score = (df["cart_trend"]*0.45 + df["time_trend"]*0.12 +
             df["discount_clicked"]*0.25 + df["returned_visit"]*0.18)
    df["will_buy"] = (score > score.median()).astype(int)

    FEATURES = ["pages_viewed","time_spent_min","items_in_cart",
                "discount_clicked","returned_visit"]
    SEQ_LEN = 7
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    X, y = [], []
    for cid in df["customer_id"].unique():
        cdf = df[df["customer_id"]==cid].reset_index(drop=True)
        for i in range(SEQ_LEN, len(cdf)):
            X.append(cdf[FEATURES].iloc[i-SEQ_LEN:i].values)
            y.append(cdf["will_buy"].iloc[i])
    X, y = np.array(X), np.array(y)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = Sequential([
        LSTM(128, input_shape=(7, 5), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=64,
              validation_split=0.1, verbose=1,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved.")
    return model, scaler

# Train if model doesn't exist
if not os.path.exists(MODEL_PATH):
    model, scaler = train()
else:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("Loaded existing model.")

# Flask API
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence'"}), 400
    sequence = np.array(data["sequence"])
    if sequence.shape != (7, 5):
        return jsonify({"error": f"Expected (7,5), got {sequence.shape}"}), 400
    seq_scaled = scaler.transform(sequence).reshape(1, 7, 5)
    prob = float(model.predict(seq_scaled, verbose=0)[0][0])
    return jsonify({
        "will_buy_probability": round(prob, 4),
        "prediction": "Yes" if prob >= 0.5 else "No",
        "confidence": f"{round(prob*100,1)}%"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
