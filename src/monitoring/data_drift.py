import numpy as np
import pandas as pd


def compute_psi(expected, actual, bins=10):

    expected_percents, bin_edges = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents)
        * np.log((actual_percents + 1e-6) / (expected_percents + 1e-6))
    )

    return psi


def detect_data_drift(train_df: pd.DataFrame, prod_df: pd.DataFrame):

    drift_report = {}

    for col in train_df.columns:

        if train_df[col].dtype != "object":

            psi = compute_psi(
                train_df[col].values,
                prod_df[col].values
            )

            drift_report[col] = psi

    return drift_report