from pydantic import BaseModel
from typing import Optional


class ScenarioInput(BaseModel):
    # статические признаки
    coolant_type: str
    wall_material: str
    hydraulic_diameter: float
    channel_length: float

    # режимные признаки
    inlet_temperature: float
    pressure: float
    mass_flux: float
    heat_flux: float
    flow_velocity: float

    # сигнальные признаки
    wall_temp_mean: Optional[float]
    wall_temp_gradient: Optional[float]
    acoustic_rms: Optional[float]
    peak_frequency: Optional[float]
    bubble_detachment_freq: Optional[float]
    vapor_area_ratio: Optional[float]

    # служебные
    solver_residual: Optional[float]
    convergence_iterations: Optional[int]