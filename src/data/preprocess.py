# src/data/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional, Tuple, Union


class Preprocessor:
    """
    Универсальный препроцессор для данных теплообмена
    Объединяет: очистку, импутацию, кодирование, масштабирование
    """

    def __init__(self):
        self.numeric_cols = None
        self.categorical_cols = None
        self.target_col = None
        self.preprocessor = None
        self.feature_names = None

        # Для отслеживания невалидных сценариев
        self.valid_ranges = {
            'pressure': (0, 500),        # давление > 0
            'mass_flux': (0, 10000),      # массовый расход > 0
            'heat_flux': (0, 5e6),        # тепловой поток >= 0
            'inlet_temperature': (0, 1200), # температура входа
            'hydraulic_diameter': (0, 1)   # гидравлический диаметр
        }

    def _remove_invalid_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаление физически невалидных сценариев
        """
        df_clean = df.copy()
        initial_len = len(df_clean)

        for col, (lower, upper) in self.valid_ranges.items():
            if col in df_clean.columns:
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

        removed = initial_len - len(df_clean)
        if removed > 0:
            print(f"  Удалено {removed} физически невалидных сценариев")

        return df_clean

    def fit(self,
            df: pd.DataFrame,
            target_col: str = 'regime_label',
            categorical_cols: Optional[List[str]] = None,
            numeric_cols: Optional[List[str]] = None,
            remove_invalid: bool = True):
        """
        Обучение препроцессора на тренировочных данных
        """
        # Определяем целевую переменную
        self.target_col = target_col

        # Удаляем невалидные сценарии если нужно
        if remove_invalid:
            df = self._remove_invalid_scenarios(df)

        # Разделяем признаки и целевую переменную
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df.copy()
            y = None

        # Автоопределение типов колонок, если не указаны явно
        if numeric_cols is None:
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_cols = numeric_cols

        if categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.categorical_cols = categorical_cols

        # Создаем пайплайн предобработки
        transformers = []

        # Числовые признаки: импутация + масштабирование
        if self.numeric_cols:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, self.numeric_cols))

        # Категориальные признаки: импутация + one-hot encoding
        if self.categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        # Обучаем на данных
        self.preprocessor.fit(X)

        # Сохраняем названия признаков после трансформации
        self.feature_names = self._get_feature_names()

        return self

    def transform(self, df: pd.DataFrame, remove_invalid: bool = True) -> np.ndarray:
        """
        Применение предобработки к данным
        """
        X = df.copy()

        # Удаляем невалидные сценарии
        if remove_invalid:
            X = self._remove_invalid_scenarios(X)

        # Если есть целевая переменная, убираем её
        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])

        # Применяем препроцессор
        X_processed = self.preprocessor.transform(X)

        return X_processed

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """Fit и transform за один шаг"""
        self.fit(df, **kwargs)
        return self.transform(df)

    def _get_feature_names(self) -> List[str]:
        """
        Получение названий признаков после трансформации
        """
        feature_names = []

        for name, transformer, cols in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                # Для one-hot получаем названия категорий
                onehot = transformer.named_steps['onehot']
                if hasattr(onehot, 'get_feature_names_out'):
                    cat_features = onehot.get_feature_names_out(cols)
                    feature_names.extend(cat_features)
                else:
                    # fallback
                    for col in cols:
                        categories = onehot.categories_[cols.index(col)]
                        feature_names.extend([f"{col}_{cat}" for cat in categories])

        return feature_names

    def get_preprocessing_report(self, df: pd.DataFrame) -> Dict:
        """
        Отчет по предобработке данных
        """
        report = {}

        # Статистика по пропускам
        missing = df.isna().sum()
        report['missing_values'] = {
            col: int(missing[col]) for col in df.columns if missing[col] > 0
        }

        # Статистика по невалидным сценариям
        invalid_counts = {}
        for col, (lower, upper) in self.valid_ranges.items():
            if col in df.columns:
                n_invalid = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_invalid > 0:
                    invalid_counts[col] = int(n_invalid)

        report['invalid_scenarios'] = invalid_counts

        # Типы данных
        report['dtypes'] = df.dtypes.astype(str).to_dict()

        return report