import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import String, Float, Integer, ForeignKey, func, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from backend.app.core.enums import (
    SessionStatus,
    SourceType,
    FatigueLevel,
    EventType,
    SeverityLevel,
)


class Base(DeclarativeBase):
    pass


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        server_default=func.now()
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status", values_callable=lambda x: [e.value for e in x]), 
        default=SessionStatus.PENDING
    )
    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType, name="source_type", values_callable=lambda x: [e.value for e in x]), 
        default=SourceType.FILE
    )
    source_name: Mapped[str | None] = mapped_column(String(255))
    duration_sec: Mapped[float | None] = mapped_column(Float)
    frames_analyzed: Mapped[int | None] = mapped_column(Integer)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    overall_fatigue_level: Mapped[FatigueLevel | None] = mapped_column(
        Enum(FatigueLevel, name="fatigue_level", values_callable=lambda x: [e.value for e in x])
    )
    error: Mapped[str | None] = mapped_column(String(1000))

    # Relationships
    events: Mapped[list["FatigueEventDB"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    frame_results: Mapped[list["FrameResultDB"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class FatigueEventDB(Base):
    __tablename__ = "fatigue_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
    default=func.now(),
    server_default=func.now()
)
    event_type: Mapped[EventType] = mapped_column(
        Enum(EventType, name="event_type", values_callable=lambda x: [e.value for e in x]), index=True
    )
    timestamp_sec: Mapped[float] = mapped_column(Float, index=True)
    duration_sec: Mapped[float | None] = mapped_column(Float)
    severity: Mapped[SeverityLevel] = mapped_column(
        Enum(SeverityLevel, name="severity", values_callable=lambda x: [e.value for e in x])
    )

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    # Relationships
    session: Mapped["AnalysisSession"] = relationship(back_populates="events")


class FrameResultDB(Base):
    __tablename__ = "frame_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True
    )
    frame_index: Mapped[int] = mapped_column(Integer, index=True)
    timestamp_sec: Mapped[float] = mapped_column(Float, index=True)
    fatigue_level: Mapped[FatigueLevel | None] = mapped_column(
        Enum(FatigueLevel, name="fatigue_level", values_callable=lambda x: [e.value for e in x])
    )
    perclos: Mapped[float | None] = mapped_column(Float)
    blink_rate: Mapped[float | None] = mapped_column(Float)
    yawn_count: Mapped[int | None] = mapped_column(Integer)
    raw_features: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Relationships
    session: Mapped["AnalysisSession"] = relationship(back_populates="frame_results")