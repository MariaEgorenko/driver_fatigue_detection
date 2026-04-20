from pydantic import BaseModel
from typing import Literal


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    db_connected: bool
