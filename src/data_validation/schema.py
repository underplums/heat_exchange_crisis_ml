from pydantic import BaseModel, Field


class SensorInput(BaseModel):

    temperature_inlet: float = Field(..., ge=0, le=1200)

    temperature_outlet: float = Field(..., ge=0, le=1200)

    heat_flux: float = Field(..., ge=0)

    flow_rate: float = Field(..., gt=0)

    pressure: float = Field(..., gt=0)

    wall_temperature: float = Field(..., ge=0, le=1500)

    time_since_start: float = Field(..., ge=0)