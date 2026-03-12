# src/data_validation/anomaly_detector.py
from sklearn.ensemble import IsolationForest
import numpy as np


class AnomalyDetector:

    def __init__(self, contamination=0.02):
        """
        Детектор аномалий на основе Isolation Forest

        Parameters:
        -----------
        contamination : float, default=0.02
            Ожидаемая доля аномалий в данных (0.02 = 2%)
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.is_fitted = False

    def fit(self, X):
        """
        Обучение детектора аномалий
        """
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Предсказание аномалий
        Returns: 1 - норма, -1 - аномалия
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")
        return self.model.predict(X)