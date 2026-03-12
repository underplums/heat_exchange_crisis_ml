# src/pipelines/feature_pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from src.data.preprocess import Preprocessor
from src.data.feature_engineering import create_features, get_feature_groups


class FeaturePipeline:
    """
    Полный пайплайн обработки признаков:
    1. Preprocessing (imputing, scaling, encoding)
    2. Feature Engineering (физические признаки)
    """

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.feature_groups = get_feature_groups()
        self.is_fitted = False

    def fit(self,
            df: pd.DataFrame,
            target_col: str = 'regime_label',
            categorical_cols: Optional[List[str]] = None):
        """
        Обучение пайплайна на тренировочных данных
        """
        # Сначала создаем инженерные признаки
        df_with_features = create_features(df)

        # Обучаем препроцессор
        self.preprocessor.fit(
            df_with_features,
            target_col=target_col,
            categorical_cols=categorical_cols,
            remove_invalid=True
        )

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Трансформация данных
        """
        if not self.is_fitted:
            raise ValueError("FeaturePipeline не обучен. Сначала вызовите fit().")

        # Создаем инженерные признаки
        df_with_features = create_features(df)

        # Применяем препроцессор
        X_processed = self.preprocessor.transform(df_with_features, remove_invalid=True)

        return X_processed

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """Fit и transform за один шаг"""
        self.fit(df, **kwargs)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Возвращает названия признаков после обработки"""
        if not self.is_fitted:
            raise ValueError("FeaturePipeline не обучен")
        return self.preprocessor.feature_names

    def get_pipeline_summary(self) -> Dict:
        """Возвращает сводку по пайплайну"""
        return {
            'n_features_original': len(self.preprocessor.numeric_cols or []) + len(self.preprocessor.categorical_cols or []),
            'n_features_final': len(self.get_feature_names()),
            'numeric_features': self.preprocessor.numeric_cols,
            'categorical_features': self.preprocessor.categorical_cols,
            'feature_groups': self.feature_groups
        }