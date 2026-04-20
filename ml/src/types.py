from dataclasses import dataclass, field
from typing import Optional, List, Literal
import numpy as np


@dataclass
class FaceLandmarks:
    """Face landmarks from MediaPipe Face Mesh."""

    # 468 точек MediaPipe Face Mesh
    points: np.ndarray  # shape (468, 3), нормализованы [0,1]
    frame_width: int
    frame_height: int
    timestamp: float  # UNIX timestamp кадра


@dataclass
class BlinkEvent:
    """Event representing a blink detected."""

    timestamp: float
    duration_ms: float
    min_ear: float


@dataclass
class YawnEvent:
    """Event representing a yawn detected."""

    timestamp: float
    duration_sec: float
    max_mar: float


@dataclass
class AbsenceEvent:
    """Event representing face absence (driver not detected)."""

    start_time: float
    duration_sec: float


@dataclass
class HeadPose:
    """Head orientation angles in degrees."""

    pitch: float  # наклон вперёд/назад (градусы)
    yaw: float  # поворот влево/вправо (градусы)
    roll: float  # крен (градусы)


@dataclass
class FeatureWindow:
    """Sliding window containing historical data for PERCLOS calculation."""

    # Данные за скользящее окно PERCLOS_WINDOW_SEC
    ear_history: List[float] = field(default_factory=list)
    mar_history: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    head_poses: List[HeadPose] = field(default_factory=list)
    blink_events: List[BlinkEvent] = field(default_factory=list)
    yawn_events: List[YawnEvent] = field(default_factory=list)
    absence_events: List[AbsenceEvent] = field(default_factory=list)
    face_detected_flags: List[bool] = field(default_factory=list)
    eye_closures_events: List['FatigueEvent'] = field(default_factory=list)


FatigueLevel = Literal["alert", "mild_fatigue", "severe_fatigue"]


@dataclass
class FatigueEvent:
    """Specific event indicating signs of fatigue."""

    event_type: Literal["eye_closure", "yawn", "head_down", "face_absent"]
    timestamp: float
    duration_sec: Optional[float]
    severity: Literal["low", "medium", "high"]
    metadata: dict = field(default_factory=dict)


@dataclass
class FatigueScore:
    level: FatigueLevel
    confidence: float  # 0.0–1.0
    perclos: float
    blink_rate: float  # морганий в минуту
    yawn_count: int  # за последнюю минуту
    head_down: bool
    face_absent: bool
    events: List[FatigueEvent] = field(default_factory=list)


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    timestamp: float
    frame_index: int
    face_detected: bool
    fatigue_score: Optional[FatigueScore]  # None если лицо не найдено
    landmarks: Optional[FaceLandmarks] = None