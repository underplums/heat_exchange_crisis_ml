# src/models/train_model.py
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix, recall_score
import numpy as np


def build_model():
    """
    Создание стекинг модели CatBoost + XGBoost с RandomForest как мета-модель
    """
    # Базовая модель 1: CatBoost
    catboost = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=300,
        loss_function='MultiClass',
        verbose=0,
        random_seed=42
    )

    # Базовая модель 2: XGBoost
    xgboost = XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )

    # Мета-модель: RandomForest (работает с мультиклассом без проблем)
    meta_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Стекинг
    model = StackingClassifier(
        estimators=[
            ('catboost', catboost),
            ('xgboost', xgboost)
        ],
        final_estimator=meta_model,
        cv=5,
        passthrough=False,
        stack_method='auto'
    )

    return model


def build_model_fast():
    """
    Быстрая версия для тестирования
    """
    catboost = CatBoostClassifier(
        iterations=150,
        depth=5,
        learning_rate=0.05,
        loss_function='MultiClass',
        verbose=0,
        random_seed=42
    )

    xgboost = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )

    meta_model = RandomForestClassifier(
        n_estimators=80,
        random_state=42,
        n_jobs=-1
    )

    model = StackingClassifier(
        estimators=[
            ('catboost', catboost),
            ('xgboost', xgboost)
        ],
        final_estimator=meta_model,
        cv=3
    )

    return model


def train_model(X_train, y_train, fast_mode=False):
    """
    Обучение модели
    """
    if fast_mode:
        model = build_model_fast()
        print("Используется быстрая версия модели")
    else:
        model = build_model()
        print("Используется полная версия модели (CatBoost + XGBoost + RandomForest)")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Оценка модели с выводом метрик
    """
    if class_names is None:
        class_names = ['Норма (0)', 'Пузырьковое (1)', 'Предкризис (2)', 'Кризис (3)']

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # F2-score (основная метрика)
    f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')
    f2_weighted = fbeta_score(y_test, y_pred, beta=2, average='weighted')

    # F1-score
    f1 = fbeta_score(y_test, y_pred, beta=1, average='macro')

    # Recall для каждого класса
    recall_per_class = recall_score(y_test, y_pred, average=None)

    print("="*60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
    print("="*60)
    print(f"F2-score (macro): {f2:.4f} (основная метрика)")
    print(f"F2-score (weighted): {f2_weighted:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

    print("\nRecall по классам:")
    for i, recall in enumerate(recall_per_class):
        print(f"  {class_names[i]}: {recall:.4f}")

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)

    return {
        'f2': f2,
        'f2_weighted': f2_weighted,
        'f1': f1,
        'recall_per_class': recall_per_class,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def predict_with_confidence(model, X):
    """
    Предсказание с оценкой уверенности
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Уверенность = максимальная вероятность
    confidence = np.max(y_proba, axis=1)

    # Уровни уверенности
    confidence_levels = []
    for c in confidence:
        if c > 0.85:
            confidence_levels.append('high')
        elif c > 0.65:
            confidence_levels.append('medium')
        else:
            confidence_levels.append('low')

    return {
        'prediction': y_pred,
        'probabilities': y_proba,
        'confidence': confidence,
        'confidence_level': confidence_levels
    }