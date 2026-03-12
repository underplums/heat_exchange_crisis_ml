import numpy as np

def compute_confidence(probabilities):

    confidence = np.max(probabilities, axis=1)

    return confidence

def confidence_label(confidence):

    if confidence > 0.85:
        return "high"

    if confidence > 0.65:
        return "medium"

    return "low"