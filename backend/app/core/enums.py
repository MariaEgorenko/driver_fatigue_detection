from enum import Enum

class SessionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SourceType(str, Enum):
    FILE = "file"
    STREAM = "stream"

class FatigueLevel(str, Enum):
    ALERT = "alert"
    MILD_FATIGUE = "mild_fatigue"
    SEVERE_FATIGUE = "severe_fatigue"

class EventType(str, Enum):
    EYE_CLOSURE = "eye_closure"
    YAWN = "yawn"
    HEAD_DOWN = "head_down"
    FACE_ABSENT = "face_absent"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"