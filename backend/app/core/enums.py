from enum import StrEnum

class SessionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SourceType(StrEnum):
    FILE = "file"
    VIDEO = "video"
    STREAM = "stream"

class FatigueLevel(StrEnum):
    ALERT = "alert"
    MILD_FATIGUE = "mild_fatigue"
    SEVERE_FATIGUE = "severe_fatigue"

class EventType(StrEnum):
    EYE_CLOSURE = "eye_closure"
    YAWN = "yawn"
    HEAD_DOWN = "head_down"
    FACE_ABSENT = "face_absent"

class SeverityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"