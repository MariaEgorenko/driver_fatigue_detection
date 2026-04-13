from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from backend.app.core.enums import(
    EventType,
    SeverityLevel,
    FatigueLevel,
    SessionStatus,
)


class FatigueEventSchema(BaseModel):
    """Diagram of a single fatigue event."""
    event_type: EventType
    timestamp_sec: float
    duration_sec: float | None = None
    severity: SeverityLevel
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventFilter(BaseModel):
    """Request parameters for filtering API events."""
    event_type: EventType | None = None
    severity: SeverityLevel | None = None
    start_time: float | None = None
    end_time: float | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class EventsResponse(BaseModel):
    """API response containing paginated events."""
    events: list[FatigueEventSchema]
    total: int
    limit: int
    offset: int


class FatigueScoreSchema(BaseModel):
    """Fatigue metrics for a frame or time interval."""
    level: FatigueLevel
    confidence: float = Field(ge=0.0, le=1.0)
    perclos: float = Field(ge=0.0, le=1.0)
    blink_rate: float = Field(ge=0.0)
    yawn_count: int = Field(ge=0)
    head_down: bool
    face_absent: bool
    events: list[FatigueEventSchema] = Field(default_factory=list)


class FaceLandmarksSchema(BaseModel):
    """Coordinates of key points of the face (for example, from MediaPipe)."""
    points: list[list[float]]
    frame_width: int
    frame_height: int


class FrameAnalysisResponse(BaseModel):
    """The result of analyzing a single frame."""
    session_id: UUID
    timestamp: float
    frame_index: int
    face_detected: bool
    fatigue_score: FatigueScoreSchema
    landmarks: FaceLandmarksSchema | None = None


class AnalysisSessionStatus(BaseModel):
    """Current status of the video stream analysis session."""
    session_id: UUID = Field(validation_alias="id")
    status: SessionStatus
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    created_at: datetime
    estimated_completion: datetime | None = None
    overall_fatigue_level: FatigueLevel | None = None
    frames_analyzed: int | None = None
    duration_sec: float | None = None
    error: str | None = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class VideoAnalysisResult(BaseModel):
    """The final result of analyzing an entire video file or a completed session."""
    session_id: UUID = Field(validation_alias="id")
    status: SessionStatus
    duration_sec: float | None = None
    overall_fatigue_level: FatigueLevel | None = None
    frames_analyzed: int = 0
    events: list[FatigueEventSchema] = Field(default_factory=list)
    timeline: list[FatigueScoreSchema] = Field(default_factory=list)
    error: str | None = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
