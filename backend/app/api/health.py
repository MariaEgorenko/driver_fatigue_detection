import logging
from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from backend.app.core.database import engine

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool


@router.get("", response_model=HealthResponse)
async def health_check(request: Request, response: Response):
    """
    Health check endpoint for orchestrators (Docker/K8s).

    Checks:
    - ML pipeline loaded (app.state.ml_pipeline)
    - Database connection (SELECT 1)
    """
    model_loaded = False
    db_connected = False

    try:
        if hasattr(request.app, "state") and hasattr(request.app.state, "ml_pipeline"):
            model_loaded = request.app.state.ml_pipeline is not None
    except Exception as e:
        logger.error(f"HealthCheck: Error checking ML pipeline state: {e}")

    try:
        if engine is not None:
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                db_connected = result.scalar() == 1
    except Exception as e:
        logger.error(f"HealthCheck: Database connection failed: {e}")

    is_healthy = model_loaded and db_connected
    
    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return HealthResponse(
        status=overall_status,
        model_loaded=model_loaded,
        db_connected=db_connected
    )