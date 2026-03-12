from pydantic import BaseModel
from typing import Optional, List


class ScenarioRequest(BaseModel):

    coolant_type: str
    wall_material: str
    hydraulic_diameter: float
    channel_length: float

    inlet_temperature: float
    pressure: float
    mass_flux: float
    heat_flux: float
    flow_velocity: float

    wall_temp_mean: Optional[float] = None
    wall_temp_gradient: Optional[float] = None
    acoustic_rms: Optional[float] = None
    peak_frequency: Optional[float] = None
    bubble_detachment_freq: Optional[float] = None
    vapor_area_ratio: Optional[float] = None

    solver_residual: Optional[float] = None
    convergence_iterations: Optional[int] = None


class PredictionResponse(BaseModel):

    predicted_regime: int
    class_probabilities: List[float]

    confidence: float
    confidence_level: str

    within_applicability_domain: bool