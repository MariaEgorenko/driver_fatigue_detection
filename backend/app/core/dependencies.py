import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request

from ml.src.inference.video_pipeline import VideoPipeline
from backend.app.core.config import get_settings
from backend.app.core.model_loader import load_model_from_registry
from backend.app.core.database import init_db, close_db

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup:
    - Load ML pipeline from MLflow Registry or file or create default
    - Initialize database connection pool

    Shutdown:
    - Cleanup resources
    """
    logger.info("Starting up application: Loading ML pipeline and DB...")

    # 1. Load ML Pipeline
    app.state.ml_pipeline = load_model_from_registry()

    # 2. Initialize Database
    try:
        await init_db()
        logger.info("Database pool initialized.")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")

    logger.info("Application startup complete.")

    yield

    # Shutdown
    logger.info("Shutting down application: Cleaning up resources...")
    try:
        await close_db()
        logger.info("Database pool closed.")
    except Exception as e:
        logger.error(f"Error while closing database: {e}")

    if hasattr(app.state, "ml_pipeline") and hasattr(app.state.ml_pipeline, "close"):
        app.state.ml_pipeline.close()

def get_ml_pipeline(request: Request) -> VideoPipeline:
    """Dependency: Возвращает ML-пайплайн из state приложения."""
    return request.app.state.ml_pipeline