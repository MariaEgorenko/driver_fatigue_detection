import logging
from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text
import asyncio

from backend.app.schemas.health import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


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
        model_loaded = (
            hasattr(request.app.state, "ml_pipeline")
            and request.app.state.ml_pipeline is not None
        )
    except Exception as e:
        logger.error(f"HealthCheck: Error checking ML pipeline state: {e}")

    db_connected = False
    try:
        engine = getattr(request.app.state, "engine", None)

        if engine is not None:
            async def _check_db():
                async with engine.connect() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    return result.scalar() == 1
                
            db_connected = await asyncio.wait_for(_check_db(), timeout=3.0)
        else:
            logger.error("HealthCheck: Engine not found in app.state")

    except asyncio.TimeoutError:
        logger.error("HealthCheck: Database connection timed out")
    except Exception as e:
        logger.error(f"HealthCheck: Database connection failed: {e}")

    is_healthy = model_loaded and db_connected
    
    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if is_healthy else "degraded",
        model_loaded=model_loaded,
        db_connected=db_connected
    )
