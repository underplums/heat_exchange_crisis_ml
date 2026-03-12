# src/data/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание физически-обоснованных инженерных признаков

    Parameters:
    -----------
    df : pd.DataFrame
        Исходные данные с сырыми признаками

    Returns:
    --------
    pd.DataFrame
        Данные с добавленными инженерными признаками
    """
    df_features = df.copy()

    # =========================================================================
    # ТЕПЛОВЫЕ ПРИЗНАКИ (Thermal features)
    # =========================================================================

    # Температурный градиент
    if all(col in df_features.columns for col in ['wall_temp_mean', 'inlet_temperature']):
        df_features['temp_gradient'] = df_features['wall_temp_mean'] - df_features['inlet_temperature']

    # Температурный напор (разница между стенкой и теплоносителем)
    if all(col in df_features.columns for col in ['wall_temp_mean', 'inlet_temperature']):
        df_features['delta_t_wall_coolant'] = df_features['wall_temp_mean'] - df_features['inlet_temperature']

    # Коэффициент теплоотдачи (грубая оценка)
    if all(col in df_features.columns for col in ['heat_flux', 'wall_temp_mean', 'inlet_temperature']):
        df_features['h_approx'] = df_features['heat_flux'] / (df_features['wall_temp_mean'] - df_features['inlet_temperature'] + 1e-6)

    # Тепловая инерция
    if all(col in df_features.columns for col in ['heat_flux', 'flow_velocity']):
        df_features['thermal_inertia'] = df_features['heat_flux'] * df_features['flow_velocity']

    # =========================================================================
    # ГИДРОДИНАМИЧЕСКИЕ ПРИЗНАКИ (Hydrodynamic features)
    # =========================================================================

    # Число Рейнольдса (прокси, без вязкости)
    if all(col in df_features.columns for col in ['mass_flux', 'hydraulic_diameter']):
        # proxy Re ~ G * D (предполагаем постоянную вязкость)
        df_features['Re_proxy'] = df_features['mass_flux'] * df_features['hydraulic_diameter']

    # Гидравлическое сопротивление (прокси)
    if all(col in df_features.columns for col in ['pressure', 'flow_velocity']):
        df_features['flow_resistance_proxy'] = df_features['pressure'] / (df_features['flow_velocity'] + 1e-6)

    # =========================================================================
    # ИНТЕГРАЛЬНЫЕ ПРИЗНАКИ (Integral features)
    # =========================================================================

    # time_to_fluence - интегральный тепловой эффект (флакс * время)
    if all(col in df_features.columns for col in ['heat_flux', 'convergence_iterations']):
        # convergence_iterations как прокси времени
        df_features['time_to_fluence'] = df_features['heat_flux'] * df_features['convergence_iterations']
    elif 'heat_flux' in df_features.columns:
        # fallback, если нет итераций
        df_features['time_to_fluence'] = df_features['heat_flux'] * 1.0

    # =========================================================================
    # АКУСТИЧЕСКИЕ ПРИЗНАКИ (Acoustic features)
    # =========================================================================

    # Нормированная амплитуда
    if all(col in df_features.columns for col in ['acoustic_rms', 'pressure']):
        df_features['norm_acoustic'] = df_features['acoustic_rms'] / (df_features['pressure'] + 1e-6)

    # Энергия акустического сигнала (прокси)
    if 'acoustic_rms' in df_features.columns:
        df_features['acoustic_energy'] = df_features['acoustic_rms'] ** 2

    # =========================================================================
    # ПАРОВЫЕ ПРИЗНАКИ (Vapor features)
    # =========================================================================

    # Паросодержание с учетом теплового потока
    if all(col in df_features.columns for col in ['vapor_area_ratio', 'heat_flux']):
        df_features['vapor_heat_ratio'] = df_features['vapor_area_ratio'] * df_features['heat_flux'] / 1e6

    # Частота отрыва пузырей (нормированная)
    if all(col in df_features.columns for col in ['bubble_detachment_freq', 'flow_velocity']):
        df_features['bubble_strouhal'] = df_features['bubble_detachment_freq'] / (df_features['flow_velocity'] + 1e-6)

    # =========================================================================
    # РЕЖИМНЫЕ ПРИЗНАКИ (Regime features)
    # =========================================================================

    # Запас до кризиса (прокси на основе параметров)
    if all(col in df_features.columns for col in ['heat_flux', 'pressure', 'inlet_temperature']):
        # Эмпирическая формула для демонстрации
        df_features['margin_to_critical'] = (
            1.0 - (df_features['heat_flux'] / 2e6) / (df_features['pressure'] / 10 + 1)
        )
        df_features['margin_to_critical'] = df_features['margin_to_critical'].clip(0, 1)

    # =========================================================================
    # ВЗАИМОДЕЙСТВИЯ ПРИЗНАКОВ (Feature interactions)
    # =========================================================================

    # Тепло + гидравлика
    if all(col in df_features.columns for col in ['heat_flux', 'mass_flux']):
        df_features['heat_mass_ratio'] = df_features['heat_flux'] / (df_features['mass_flux'] + 1e-6)

    # Давление + температура
    if all(col in df_features.columns for col in ['pressure', 'inlet_temperature']):
        df_features['p_t_product'] = df_features['pressure'] * df_features['inlet_temperature'] / 1000

    return df_features


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Возвращает словарь с группировкой признаков по типам
    """
    return {
        'thermal': ['heat_flux', 'wall_temp_mean', 'wall_temp_gradient', 'temp_gradient',
                    'delta_t_wall_coolant', 'h_approx', 'thermal_inertia'],

        'hydrodynamic': ['pressure', 'mass_flux', 'flow_velocity', 'Re_proxy',
                         'flow_resistance_proxy', 'heat_mass_ratio'],

        'acoustic': ['acoustic_rms', 'peak_frequency', 'norm_acoustic', 'acoustic_energy'],

        'vapor': ['vapor_area_ratio', 'bubble_detachment_freq', 'vapor_heat_ratio', 'bubble_strouhal'],

        'integral': ['time_to_fluence', 'margin_to_critical', 'p_t_product'],

        'static': ['coolant_type', 'wall_material', 'hydraulic_diameter', 'channel_length'],

        'regime': ['inlet_temperature', 'pressure', 'mass_flux', 'heat_flux', 'flow_velocity']
    }


def get_feature_importance_ranking() -> Dict[str, float]:
    """
    Возвращает ожидаемую важность признаков (для демонстрации)
    """
    return {
        'heat_flux': 0.15,
        'pressure': 0.12,
        'vapor_area_ratio': 0.10,
        'temp_gradient': 0.09,
        'mass_flux': 0.08,
        'margin_to_critical': 0.08,
        'time_to_fluence': 0.07,
        'acoustic_rms': 0.06,
        'Re_proxy': 0.05,
        'wall_temp_mean': 0.05,
        'bubble_detachment_freq': 0.05,
        'hydraulic_diameter': 0.04,
        'coolant_type': 0.03,
        'wall_material': 0.03
    }