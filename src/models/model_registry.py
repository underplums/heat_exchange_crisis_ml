# src/models/model_registry.py
import joblib
import json
import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np


class ModelRegistry:
    """
    Реестр моделей для сохранения и загрузки
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)

    def save_model(self,
                   model: Any,
                   model_name: str = "boiling_model",
                   metadata: Optional[Dict] = None) -> str:
        """
        Сохранение модели
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{model_name}_{timestamp}.pkl"

        # Сохраняем модель
        joblib.dump(model, model_path)
        print(f"✓ Модель сохранена: {model_path}")

        # Сохраняем метаданные если есть
        if metadata:
            meta_path = self.model_dir / f"{model_name}_{timestamp}_metadata.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=self._convert)

        return str(model_path)

    def load_model(self, model_path: Optional[str] = None, model_name: str = "boiling_model") -> Any:
        """
        Загрузка модели
        """
        if model_path:
            path = Path(model_path)
        else:
            # Ищем последнюю модель
            models = sorted(self.model_dir.glob(f"{model_name}_*.pkl"))
            if not models:
                raise FileNotFoundError(f"Модель {model_name} не найдена")
            path = models[-1]

        print(f"Загрузка модели: {path}")
        return joblib.load(path)

    def list_models(self, model_name: str = "boiling_model") -> pd.DataFrame:
        """
        Список моделей
        """
        models = []
        for path in sorted(self.model_dir.glob(f"{model_name}_*.pkl")):
            models.append({
                'name': path.name,
                'path': str(path),
                'size_mb': round(path.stat().st_size / (1024*1024), 2),
                'created': datetime.datetime.fromtimestamp(path.stat().st_ctime).strftime('%Y-%m-%d %H:%M')
            })
        return pd.DataFrame(models)

    def _convert(self, obj):
        """Конвертация для JSON"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Явно указываем, что экспортируем
__all__ = ['ModelRegistry']


# Для обратной совместимости с простыми функциями
def save_model(model, model_name="boiling_model", metadata=None):
    registry = ModelRegistry()
    return registry.save_model(model, model_name, metadata)


def load_model(model_path=None, model_name="boiling_model"):
    registry = ModelRegistry()
    return registry.load_model(model_path, model_name)


def list_models(model_name="boiling_model"):
    registry = ModelRegistry()
    return registry.list_models(model_name)