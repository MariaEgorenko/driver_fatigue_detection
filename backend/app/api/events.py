import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.schemas.analysis import EventFilter, EventsResponse, FatigueEventSchema
from backend.app.models.db_models import FatigueEventDB, AnalysisSession
from backend.app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get(
    "/session/{session_id}",
    response_model=EventsResponse,
    summary="Get fatigue events for a session",
    description="Retrieve fatigue events filtered by session_id with optional filtering by event type, severity, and time range.",
)
async def get_events(
    session_id: UUID, 
    filters: EventFilter = Depends(), 
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve fatigue events for a specific analysis session.
    """
    try:
        session_result = await db.get(AnalysisSession, session_id)
        if not session_result:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )

        conditions = [FatigueEventDB.session_id == session_id]

        if filters.event_type:
            conditions.append(FatigueEventDB.event_type == filters.event_type)

        if filters.severity:
            conditions.append(FatigueEventDB.severity == filters.severity)

        if filters.start_time is not None:
            conditions.append(FatigueEventDB.timestamp_sec >= filters.start_time)

        if filters.end_time is not None:
            conditions.append(FatigueEventDB.timestamp_sec <= filters.end_time)

        count_stmt = select(func.count()).select_from(FatigueEventDB).where(*conditions)
        total = await db.scalar(count_stmt) or 0

        stmt = (
            select(FatigueEventDB)
            .where(*conditions)
            .order_by(FatigueEventDB.timestamp_sec.asc())
            .limit(filters.limit)
            .offset(filters.offset)
        )
        result = await db.scalars(stmt)
        events = result.all()

        event_schemas = [
            FatigueEventSchema(
                event_type=event.event_type,
                timestamp_sec=event.timestamp_sec,
                duration_sec=event.duration_sec,
                severity=event.severity,
                metadata=event.metadata_, 
            )
            for event in events
        ]

        return EventsResponse(
            events=event_schemas,
            total=total,
            limit=filters.limit,
            offset=filters.offset,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving events for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving events",
        )