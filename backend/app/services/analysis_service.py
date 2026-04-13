import uuid
import logging
from typing import Any

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.db_models import AnalysisSession, FatigueEventDB, FrameResultDB
from backend.app.core.enums import SessionStatus, SourceType

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for managing video analysis sessions."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create_session(
        self,
        source_type: SourceType | str,
        source_name: str | None = None,
    ) -> AnalysisSession:
        """Create a new analysis session."""
        
        if source_type == "video":
            source_type = SourceType.FILE

        session = AnalysisSession(
            status=SessionStatus.PENDING,
            source_type=source_type,
            source_name=source_name,
        )
        self._db.add(session)
        await self._db.commit()
        await self._db.refresh(session)
        return session

    async def update_session(
        self,
        session_id: uuid.UUID | str,
        status: SessionStatus | str,
        progress: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Update session status, progress, and other fields."""
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)

        values: dict[str, Any] = {"status": status}
        if progress is not None:
            values["progress"] = progress
        values.update(kwargs)
        
        await self._db.execute(
            update(AnalysisSession)
            .where(AnalysisSession.id == session_id)
            .values(**values)
        )
        await self._db.commit()

    async def save_events(
        self,
        session_id: uuid.UUID | str,
        events: list[dict[str, Any]],
    ) -> None:
        """Batch insert fatigue events."""
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)

        db_events = []
        for e in events:
            event_data = e.copy()
            if "metadata" in event_data:
                event_data["metadata_"] = event_data.pop("metadata")
            
            db_events.append(FatigueEventDB(session_id=session_id, **event_data))

        self._db.add_all(db_events)
        await self._db.commit()

    async def save_frame_results(
        self,
        session_id: uuid.UUID | str,
        results: list[dict[str, Any]],
    ) -> None:
        """Batch insert frame results (every FRAME_SAVE_INTERVAL frames)."""
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)

        db_results = [FrameResultDB(session_id=session_id, **r) for r in results]
        self._db.add_all(db_results)
        await self._db.commit()

    async def get_session(self, session_id: uuid.UUID | str) -> AnalysisSession | None:
        """Get session by ID with relationships loaded."""
        if isinstance(session_id, str):
            try:
                session_id = uuid.UUID(session_id)
            except ValueError:
                return None

        result = await self._db.execute(
            select(AnalysisSession)
            .where(AnalysisSession.id == session_id)
            .options(
                selectinload(AnalysisSession.events), 
                selectinload(AnalysisSession.frame_results)
            )
        )
        return result.scalar_one_or_none()

    async def list_sessions(
        self, page: int = 1, limit: int = 20
    ) -> tuple[list[AnalysisSession], int]:
        """List sessions with pagination."""
        
        count_query = select(func.count()).select_from(AnalysisSession)
        total = await self._db.scalar(count_query) or 0

        offset = (page - 1) * limit
        result = await self._db.execute(
            select(AnalysisSession)
            .offset(offset)
            .limit(limit)
            .order_by(AnalysisSession.created_at.desc())
        )
        sessions = result.scalars().all()
        
        return list(sessions), total
