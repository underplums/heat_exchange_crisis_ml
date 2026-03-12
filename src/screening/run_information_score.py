import numpy as np


class InformationScorer:

    def __init__(self):
        pass

    # -------------------------
    # MODEL UNCERTAINTY
    # -------------------------

    def uncertainty_score(self, probabilities):
        return 1 - np.max(probabilities)

    # -------------------------
    # NOVELTY
    # -------------------------

    def novelty_score(self, distance_to_train):
        return distance_to_train

    # -------------------------
    # REGIME IMPORTANCE
    # -------------------------

    def regime_importance(self, predicted_class):

        weights = {
            0: 0.2,  # normal
            1: 0.4,  # transitional
            2: 0.8,  # precritical
            3: 1.0   # critical
        }

        return weights.get(predicted_class, 0.5)

    # -------------------------
    # FINAL PRIORITY SCORE
    # -------------------------

    def compute_priority_score(
        self,
        probabilities,
        distance_to_train,
        predicted_class
    ):

        uncertainty = self.uncertainty_score(probabilities)

        novelty = self.novelty_score(distance_to_train)

        importance = self.regime_importance(predicted_class)

        score = uncertainty + novelty + importance

        return {
            "uncertainty": uncertainty,
            "novelty": novelty,
            "regime_importance": importance,
            "priority_score": score
        }

    # -------------------------
    # RANK SCENARIOS
    # -------------------------

    def rank_scenarios(self, scenarios, top_k=None):

        ranked = sorted(
            scenarios,
            key=lambda x: x["priority_score"],
            reverse=True
        )

        if top_k:
            return ranked[:top_k]

        return ranked