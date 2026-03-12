# src/pipelines/train_pipeline.py
import pandas as pd
import numpy as np
import joblib
import datetime
import os
from typing import Dict, Optional, Tuple

from src.data_validation.anomaly_detector import AnomalyDetector
from src.data.data_splitter import DataSplitter
from src.pipelines.feature_pipeline import FeaturePipeline
from src.models.train_model import build_model, evaluate_model
from src.models.model_registry import ModelRegistry  # Теперь должно работать


class TrainPipeline:
    """
    Полный пайплайн обучения модели
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.anomaly_detector = None
        self.data_splitter = None
        self.feature_pipeline = None
        self.model = None
        self.metrics = {}

    def run(self,
            data_path: str,
            target_col: str = 'regime_label',
            test_size: float = 0.2,
            val_size: float = 0.2,
            contamination: float = 0.05,
            save_model: bool = True) -> Dict:

        print("=" * 60)
        print("ЗАПУСК TRAIN PIPELINE")
        print("=" * 60)

        # ШАГ 1: Загрузка данных
        print("\n[1] Загрузка данных...")
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        print(f"    Загружено {len(df)} сценариев")

        # ШАГ 2: Anomaly Detection
        print("\n[2] Anomaly Detection...")
        numeric_features = ['heat_flux', 'pressure', 'mass_flux', 'wall_temp_mean']
        available_features = [f for f in numeric_features if f in df.columns]
        X_numeric = df[available_features].values

        self.anomaly_detector = AnomalyDetector(contamination=contamination)
        self.anomaly_detector.fit(X_numeric)
        anomaly_mask = self.anomaly_detector.predict(X_numeric) == -1
        print(f"    Аномалий: {anomaly_mask.sum()} ({100*anomaly_mask.sum()/len(df):.1f}%)")

        # ШАГ 3: Split
        print("\n[3] Разделение данных...")
        self.data_splitter = DataSplitter(
            test_size=test_size,
            val_size=val_size,
            random_state=self.random_state,
            group_col='wall_material'
        )

        train_df, val_df, test_df = self.data_splitter.split(
            df, target_col=target_col, anomaly_mask=anomaly_mask
        )

        print(f"    Train: {len(train_df)}")
        print(f"    Val: {len(val_df)}")
        print(f"    Test: {len(test_df)}")

        # ШАГ 4: Feature Pipeline
        print("\n[4] Feature Engineering...")
        self.feature_pipeline = FeaturePipeline()

        X_train = self.feature_pipeline.fit_transform(
            train_df, target_col=target_col,
            categorical_cols=['coolant_type', 'wall_material']
        )
        X_val = self.feature_pipeline.transform(val_df)
        X_test = self.feature_pipeline.transform(test_df)

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        print(f"    Признаков после обработки: {X_train.shape[1]}")

        # ШАГ 5: Обучение
        print("\n[5] Обучение модели...")
        self.model = build_model()
        self.model.fit(X_train, y_train)

        # ШАГ 6: Оценка
        print("\n[6] Оценка модели...")
        from sklearn.metrics import fbeta_score, recall_score

        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)

        self.metrics = {
            'f2_val': float(fbeta_score(y_val, y_val_pred, beta=2, average='macro')),
            'f2_test': float(fbeta_score(y_test, y_test_pred, beta=2, average='macro')),
            'recall_val': float(recall_score(y_val, y_val_pred, average='macro')),
            'recall_test': float(recall_score(y_test, y_test_pred, average='macro')),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }

        print(f"    F2-val: {self.metrics['f2_val']:.4f}")
        print(f"    F2-test: {self.metrics['f2_test']:.4f}")

        # ШАГ 7: Сохранение
        if save_model:
            print("\n[7] Сохранение модели...")
            registry = ModelRegistry()
            model_path = registry.save_model(
                self.model,
                model_name="boiling_model",
                metadata={'metrics': self.metrics}
            )
            print(f"    Модель сохранена: {model_path}")

        print("\n" + "=" * 60)
        print("TRAIN PIPELINE ЗАВЕРШЕН")
        print("=" * 60)

        return {
            'model': self.model,
            'metrics': self.metrics,
            'feature_pipeline': self.feature_pipeline,
            'anomaly_detector': self.anomaly_detector
        }


# Упрощенная функция для запуска
def run_training_pipeline(data_path: str, **kwargs):
    pipeline = TrainPipeline()
    return pipeline.run(data_path, **kwargs)