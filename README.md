# Customer Purchase Prediction — LSTM

> Predicts whether a customer will make a purchase in the next 7 days based on their browsing history, using a stacked LSTM neural network.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Docker](https://img.shields.io/badge/Docker-deployed-blue)

## Live Demo
**API URL:** https://purchase-predictor-nkcx.onrender.com

> Note: First request may take 60-90 seconds (free tier cold start)

## Quick Test
    curl -X POST https://purchase-predictor-nkcx.onrender.com/predict \
      -H "Content-Type: application/json" \
      -d '{"sequence":[[3,5.2,2,0,1],[4,8.1,3,1,0],[2,3.4,3,0,1],[5,12.0,4,1,1],[3,6.7,4,0,0],[4,9.2,5,1,1],[6,14.3,5,1,1]]}'

## Model Performance
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.877 |
| Accuracy | 79% |
| Precision (buyers) | 77% |
| Recall (buyers) | 80% |

## Architecture
    Input (7 days x 5 features)
        ↓
    LSTM Layer (128 units) + Dropout 20%
        ↓
    LSTM Layer (64 units) + Dropout 20%
        ↓
    Dense (sigmoid) → Purchase Probability

## Input Features
| Feature | Description |
|---------|-------------|
| pages_viewed | Number of pages visited that day |
| time_spent_min | Minutes spent on site |
| items_in_cart | Cart size at end of day |
| discount_clicked | Clicked a coupon/offer (0 or 1) |
| returned_visit | Visited site more than once (0 or 1) |

## Project Structure
    purchase-predictor/
    ├── train_and_serve.py   — Model training + Flask API
    ├── app.py               — Flask API (local)
    ├── Dockerfile           — Container config
    ├── requirements.txt     — Dependencies
    ├── roc_curve.png        — ROC curve chart
    └── confusion_matrix.png — Confusion matrix

## Tech Stack
- **Model:** TensorFlow/Keras LSTM
- **API:** Flask REST
- **Container:** Docker
- **Hosting:** Render (free tier)

## Local Setup
    git clone https://github.com/anujburman/Purchase_Predictor
    cd Purchase_Predictor
    pip install -r requirements.txt
    python train_and_serve.py
