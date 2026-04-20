import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.dependencies import lifespan
from backend.app.api.health import router as health_router
from backend.app.api.analysis import router as analysis_router
from backend.app.api.events import router as events_router
from backend.app.core.metrics import metrics_router, MetricsMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Driver Fatigue Detection API",
    version="1.0.0",
    description="API for detecting driver fatigue using CV/ML models",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(MetricsMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(health_router, prefix="/health", tags=["System"])
app.include_router(metrics_router, tags=["System"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(events_router, prefix="/api/v1/events", tags=["Events"])