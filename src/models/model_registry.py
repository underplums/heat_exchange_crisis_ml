import joblib
from pathlib import Path
import datetime


MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(exist_ok=True)


def save_model(model):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    model_path = MODEL_DIR / f"boiling_model_{timestamp}.pkl"

    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    return model_path