import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

from src.data_validation.data_validator import DataValidator
from src.data.feature_engineering import create_features
from src.models.train_model import train_model, evaluate_model
from src.models.model_registry import save_model


def run_training_pipeline(data_path: str):

    print("Loading data...")
    df = pd.read_parquet(data_path)

    # -----------------------------
    # DATA VALIDATION
    # -----------------------------

    validator = DataValidator()
    validator.validate(df)

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------

    df = create_features(df)

    # -----------------------------
    # TARGET
    # -----------------------------

    y = df["regime_label"]

    X = df.drop(columns=["regime_label"])

    # -----------------------------
    # GROUP-AWARE SPLIT
    # -----------------------------

    groups = df["wall_material"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------

    print("Training model...")

    model = train_model(X_train, y_train)

    # -----------------------------
    # EVALUATION
    # -----------------------------

    print("Evaluating model...")

    evaluate_model(model, X_test, y_test)

    # -----------------------------
    # SAVE MODEL
    # -----------------------------

    save_model(model)

    print("Training pipeline finished.")