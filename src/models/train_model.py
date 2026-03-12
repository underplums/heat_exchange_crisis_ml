from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import fbeta_score, classification_report

"""
We used a stacked ensemble combining CatBoost and XGBoost as base learners, with a logistic regression meta-model.
CatBoost handled categorical effects well, while XGBoost captured complex nonlinear interactions.
The stacking layer improved robustness for rare regimes such as precritical boiling.
"""

def build_model():

    cat_model = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=300,
        loss_function="MultiClass",
        verbose=False
    )

    xgb_model = XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False
    )

    base_models = [
        ("catboost", cat_model),
        ("xgboost", xgb_model),
    ]

    meta_model = LogisticRegression(max_iter=1000)

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=False
    )

    return model


def train_model(X_train, y_train):

    model = build_model()

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    f2 = fbeta_score(
        y_test,
        preds,
        beta=2,
        average="macro"
    )

    print("F2 score:", f2)
    print(classification_report(y_test, preds))

    return f2