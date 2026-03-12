import joblib
import numpy as np
from pathlib import Path


MODEL_DIR = Path("models")


def load_latest_model():

    models = sorted(MODEL_DIR.glob("boiling_model_*.pkl"))

    if not models:
        raise FileNotFoundError("No trained model found")

    latest_model = models[-1]

    print(f"Loading model: {latest_model}")

    return joblib.load(latest_model)


def predict(model, X):

    preds = model.predict(X)

    probs = model.predict_proba(X)

    return preds, probs