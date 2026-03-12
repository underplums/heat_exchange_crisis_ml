from fastapi import FastAPI

from src.api.schemas import ScenarioRequest, PredictionResponse
from src.api.inference_service import InferenceService


app = FastAPI(
    title="Boiling Crisis ML Service",
    description="Surrogate ML model for boiling regime prediction",
    version="1.0"
)


service = InferenceService()


@app.get("/")
def healthcheck():

    return {"status": "service running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ScenarioRequest):

    scenario = request.dict()

    result = service.predict(scenario)

    return result



"""
Example:
{
  "coolant_type": "water",
  "wall_material": "inconel",
  "hydraulic_diameter": 0.01,
  "channel_length": 1.2,

  "inlet_temperature": 320,
  "pressure": 15,
  "mass_flux": 1200,
  "heat_flux": 850000,
  "flow_velocity": 2.3,

  "wall_temp_mean": 360,
  "acoustic_rms": 0.2,
  "peak_frequency": 4500,
  "bubble_detachment_freq": 120
}
"""