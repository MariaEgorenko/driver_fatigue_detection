import time
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import Match

metrics_router = APIRouter()

# --- ML & Business Metrics ---

model_info = Info("fatigue_model", "Loaded model information")

frame_processing_duration = Histogram(
    "fatigue_frame_processing_seconds",
    "Frame ML processing time",
    buckets=[0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0],
)

active_sessions = Gauge("fatigue_active_sessions", "Currently processing sessions")

fatigue_events_total = Counter(
    "fatigue_events_total", "Fatigue events detected", ["event_type", "severity"]
)

analysis_errors_total = Counter("fatigue_analysis_errors_total", "Analysis errors", ["error_type"])

# --- HTTP Metrics ---

requests_total = Counter(
    "fatigue_api_requests_total", "Total API requests", ["method", "endpoint", "status_code"]
)

request_duration_seconds = Histogram(
    "fatigue_api_request_duration_seconds",
    "Request duration",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        status_code = 500 

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            raise e
        finally:
            duration = time.time() - start_time
            
            endpoint = request.url.path
            for route in request.app.routes:
                match, _ = route.matches(request.scope)
                if match == Match.FULL:
                    endpoint = route.path
                    break
            
            if status_code == 404:
                endpoint = "/404_not_found"

            requests_total.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()

            request_duration_seconds.labels(endpoint=endpoint).observe(duration)


@metrics_router.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)