import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
import copy

from fastapi import FastAPI, Request

from ml.src.inference.video_pipeline import VideoPipeline
from backend.app.core.model_loader import load_model_from_registry
from backend.app.core.database import setup_db, init_models

logger = logging.getLogger(__name__)

active_pipelines: Dict[str, VideoPipeline] = {}

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

    try:
        app.state.ml_pipeline = await asyncio.to_thread(load_model_from_registry)
        logger.info("ML Pipeline loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load ML Pipeline: {e}")
        raise

    try:
        engine, db_sessionmaker = setup_db()
        app.state.engine = engine
        app.state.db_sessionmaker = db_sessionmaker
        
        await init_models(engine)
        logger.info("Database pool initialized and attached to app.state.")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")
        raise

    logger.info("Application startup complete.")

    yield

    logger.info("Shutting down application: Cleaning up resources...")
    try:
        engine = getattr(app.state, "engine", None)
        if engine:
            await engine.dispose()
            logger.info("Database pool closed.")
    except Exception as e:
        logger.error(f"Error while closing database: {e}")

    if hasattr(app.state, "ml_pipeline") and hasattr(app.state.ml_pipeline, "close"):
        try:
            app.state.ml_pipeline.close()
            logger.info("ML Pipeline resources released.")
        except Exception as e:
            logger.error(f"Error while closing ML Pipeline: {e}")

def get_ml_pipeline(request: Request) -> VideoPipeline:
    """
        Dependency: Returns a FRESH instance of the ML pipeline 
        so that requests don't mix their FeatureWindows.
    """
    base_pipeline = getattr(request.app.state, "ml_pipeline", None)
    return VideoPipeline(base_pipeline._config)

def get_streaming_pipeline(session_id: str, request: Request) -> VideoPipeline:
    """For real-time streaming frame by frame (we maintain context between requests)"""
    if session_id not in active_pipelines:
        base_pipeline = getattr(request.app.state, "ml_pipeline", None)
        active_pipelines[session_id] = VideoPipeline(base_pipeline._config)
        
    return active_pipelines[session_id]