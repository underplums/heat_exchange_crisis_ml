import pandas as pd


class DataValidator:

    def check_missing(self, df: pd.DataFrame):

        missing_ratio = df.isna().mean()

        if (missing_ratio > 0.5).any():
            raise ValueError("Too many missing values in some features")

        return True


    def check_ranges(self, df: pd.DataFrame):

        if (df["mass_flux"] <= 0).any():
            raise ValueError("Mass flux must be positive")

        if (df["pressure"] <= 0).any():
            raise ValueError("Pressure must be positive")

        return True


    def validate(self, df: pd.DataFrame):

        self.check_missing(df)
        self.check_ranges(df)

        return df