import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.schemas.analysis import EventFilter, EventsResponse
from backend.app.services.analysis_service import AnalysisService
from backend.app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_analysis_service(
    db: AsyncSession = Depends(get_db)
) -> AnalysisService:
    return AnalysisService(db)

@router.get(
    "/session/{session_id}",
    response_model=EventsResponse,
    summary="Get fatigue events for a session",
    description="Retrieve fatigue events filtered by session_id with optional filtering by event type, severity, and time range.",
)
async def get_events(
    session_id: UUID, 
    filters: EventFilter = Depends(), 
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Retrieve fatigue events for a specific analysis session.
    """
    try:
        if (
            filters.start_time is not None
            and filters.end_time is not None
            and filters.start_time > filters.end_time
        ):
            raise HTTPException(
                status_code=422,
                detail="start_time must be less than or equal to end_time",
            )
        
        result = await service.get_session_events(session_id=session_id, filters=filters)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving events for session {session_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving events",
        )
