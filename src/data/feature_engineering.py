import pandas as pd


def create_features(df: pd.DataFrame):

    # температурный градиент
    df["temp_gradient"] = (
        df["wall_temp_mean"] - df["inlet_temperature"]
    )

    # интегральный тепловой эффект
    df["heat_flux_effect"] = (
        df["heat_flux"] * df["flow_velocity"]
    )

    # proxy накопленного воздействия
    df["fluence_proxy"] = (
        df["heat_flux"] * df["convergence_iterations"]
    )

    return df