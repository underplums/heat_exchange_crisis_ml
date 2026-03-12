from sklearn.metrics import fbeta_score


def evaluate_model_drift(y_true, y_pred):

    f2 = fbeta_score(
        y_true,
        y_pred,
        beta=2,
        average="macro"
    )

    return {
        "f2_score": f2
    }