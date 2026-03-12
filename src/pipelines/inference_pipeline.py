import pandas as pd

from src.data.feature_engineering import create_features
from src.models.predict import load_latest_model, predict
from src.confidence.confidence_score import compute_confidence, confidence_label
from src.applicability.applicability_domain import ApplicabilityDomain


class InferencePipeline:

    def __init__(self):

        self.model = load_latest_model()

        self.applicability = ApplicabilityDomain()

    def predict_scenario(self, scenario: dict):

        df = pd.DataFrame([scenario])

        # -------------------
        # feature engineering
        # -------------------

        df = create_features(df)

        X = df.values

        # -------------------
        # model prediction
        # -------------------

        pred, probs = predict(self.model, X)

        confidence = compute_confidence(probs)[0]

        confidence_lvl = confidence_label(confidence)

        # -------------------
        # applicability check
        # -------------------

        in_domain = True

        try:
            in_domain = self.applicability.is_within_domain(X[0])
        except:
            pass

        result = {

            "predicted_regime": int(pred[0]),

            "class_probabilities": probs[0].tolist(),

            "confidence": float(confidence),

            "confidence_level": confidence_lvl,

            "within_applicability_domain": in_domain

        }

        return result