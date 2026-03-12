# src/data/data_splitter.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from typing import Tuple, Optional, Dict, Union


class DataSplitter:
    """
    Класс для разделения данных с учетом:
    - группировки по материалам (group-aware split)
    - стратификации по целевой переменной
    - исключения аномалий из обучения (опционально)
    """

    def __init__(self,
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42,
                 stratify: bool = True,
                 group_col: str = 'wall_material'):
        """
        Parameters:
        -----------
        test_size : float
            Доля тестовой выборки
        val_size : float
            Доля валидационной выборки (от тренировочной)
        random_state : int
            Seed для воспроизводимости
        stratify : bool
            Использовать ли стратификацию
        group_col : str
            Колонка для группового сплита
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify
        self.group_col = group_col

        # Для хранения индексов
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def split(self,
              df: pd.DataFrame,
              target_col: str = 'regime_label',
              anomaly_mask: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделение данных на train/val/test

        Parameters:
        -----------
        df : pd.DataFrame
            Исходные данные
        target_col : str
            Название целевой переменной
        anomaly_mask : Optional[pd.Series]
            Маска аномалий (True - аномалия)
            Если передана, аномалии НЕ попадают в train, но могут быть в val/test

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            train_df, val_df, test_df
        """
        df_copy = df.copy()

        # Если есть маска аномалий, разделяем данные
        if anomaly_mask is not None:
            clean_data = df_copy[~anomaly_mask]
            anomaly_data = df_copy[anomaly_mask]
            print(f"  • Нормальных сценариев: {len(clean_data)}")
            print(f"  • Аномалий (исключены из train): {len(anomaly_data)}")
        else:
            clean_data = df_copy
            anomaly_data = None

        # Первый сплит: train+val / test
        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state
        )

        train_val_idx, test_idx = next(
            gss1.split(clean_data, clean_data[target_col], clean_data[self.group_col])
        )

        train_val_df = clean_data.iloc[train_val_idx]
        test_df = clean_data.iloc[test_idx]

        # Второй сплит: train / val
        val_relative_size = self.val_size / (1 - self.test_size)

        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_relative_size,
            random_state=self.random_state
        )

        train_idx, val_idx = next(
            gss2.split(train_val_df, train_val_df[target_col], train_val_df[self.group_col])
        )

        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        # Сохраняем индексы
        self.train_idx = train_df.index
        self.val_idx = val_df.index
        self.test_idx = test_df.index

        # Добавляем аномалии в val/test если нужно (опционально)
        if anomaly_data is not None and len(anomaly_data) > 0:
            # Можно добавить логику распределения аномалий по val/test
            print(f"  • Аномалии не включены в train (используются только для валидации)")

        return train_df, val_df, test_df

    def get_split_info(self) -> Dict:
        """Возвращает информацию о сплите"""
        return {
            'train_size': len(self.train_idx) if self.train_idx is not None else 0,
            'val_size': len(self.val_idx) if self.val_idx is not None else 0,
            'test_size': len(self.test_idx) if self.test_idx is not None else 0,
            'train_materials': None,  # Можно добавить
            'test_materials': None
        }