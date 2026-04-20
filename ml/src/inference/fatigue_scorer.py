import logging
from typing import List

from ml.src.types import (
    FeatureWindow,
    FatigueScore,
    FatigueEvent,
    FatigueLevel,
)
from ml.src.features.eye_features import EyeAnalyzer
from ml.src.features.mouth_features import MouthAnalyzer
from ml.src.features.head_pose import HeadPoseEstimator

logger = logging.getLogger(__name__)

YAWN_COUNT_WINDOW_SEC = 300.0  # 5 minutes to count yawns
HEAD_DOWN_RECENT_FRAMES = 10 # how many of the last frames should I count


class FatigueScorer:
    def __init__(self, config) -> None:
        self._config = config
        self._eye = EyeAnalyzer(config)
        self._mouth = MouthAnalyzer(config)
        self._last_score_time = -100.0
        self._last_headdown_event_time = -100.0
        self._last_reported_absence_start = -100.0

    def score(self, window: FeatureWindow) -> FatigueScore:
        """Aggregate attributes in FatigueScore."""
        
        # Handle empty window
        if not window.timestamps or not window.ear_history:
            return FatigueScore(
                level="alert",
                confidence=0.0,
                perclos=0.0,
                blink_rate=0.0,
                yawn_count=0,
                head_down=False,
                face_absent=False,
                events=[],
            )

        current_time = window.timestamps[-1]

        # 1. Compute PERCLOS
        perclos = self._eye.compute_perclos(window)

        # 2. Compute blink rate
        blink_rate = self._eye.blink_rate(window)

        # 3. Count yawns in last 5 minutes
        yawn_window_start = current_time - YAWN_COUNT_WINDOW_SEC
        recent_yawns = [y for y in window.yawn_events if y.timestamp >= yawn_window_start]
        yawn_count = len(recent_yawns)

        # 4. Determine head_down
        head_down = False
        if len(window.head_poses) >= HEAD_DOWN_RECENT_FRAMES:
            recent_poses = window.head_poses[-HEAD_DOWN_RECENT_FRAMES:]
            head_down_count = sum(
                1 for pose in recent_poses 
                if pose is not None and abs(pose.pitch) > self._config.HEAD_PITCH_THRESHOLD_DEG
            )
            head_down = head_down_count >= (HEAD_DOWN_RECENT_FRAMES * 0.7)

        # 5. Determine face_absen
        face_absent = False
        is_currently_absent = bool(window.face_detected_flags and not window.face_detected_flags[-1])

        if is_currently_absent and window.absence_events:
            last_absence = window.absence_events[-1]
            if current_time - last_absence.start_time >= self._config.FACE_ABSENT_THRESHOLD_SEC:
                face_absent = True

        # 6. Build events list
        events: List[FatigueEvent] = []

        new_closures = [
            c for c in window.eye_closures_events
            if (c.timestamp + (c.duration_sec or 0)) > self._last_score_time
        ]
        events.extend(new_closures)

        new_yawns = [y for y in recent_yawns if y.timestamp > self._last_score_time]

        # Yawn events
        for yawn in new_yawns:
            severity = "high" if yawn_count >= 4 else "medium" if yawn_count >= 2 else "low"
            events.append(
                FatigueEvent(
                    event_type="yawn",
                    timestamp=yawn.timestamp,
                    duration_sec=yawn.duration_sec,
                    severity=severity,
                    metadata={"max_mar": yawn.max_mar},
                )
            )

        # Head down event
        if head_down:
            if current_time - self._last_headdown_event_time >= 2.0:
                events.append(
                    FatigueEvent(
                        event_type="head_down",
                        timestamp=current_time,
                        duration_sec=None,
                        severity="medium",
                        metadata={"pitch_threshold": self._config.HEAD_PITCH_THRESHOLD_DEG},
                    )
                )
                self._last_headdown_event_time = current_time

        # Face absent events
        new_absences = [
            a for a in window.absence_events
            if a.start_time > self._last_reported_absence_start
        ]

        for absence in new_absences:
            if absence.duration_sec >= self._config.FACE_ABSENT_THRESHOLD_SEC:
                events.append(
                    FatigueEvent(
                        event_type="face_absent",
                        timestamp=absence.start_time,
                        duration_sec=absence.duration_sec,
                        severity="high",
                        metadata={"duration": absence.duration_sec},
                    )
                )
                self._last_reported_absence_start = absence.start_time

        # 7. Determine level
        level = self._determine_level(
            perclos=perclos,
            yawn_count=yawn_count,
            head_down=head_down,
            face_absent=face_absent
        )

        self._last_score_time = current_time

        # 8. Calculate confidence
        confidence = self._calculate_confidence(
            level=level,
            perclos=perclos,
            yawn_count=yawn_count,
            head_down=head_down,
            face_absent=face_absent,
            window_size=len(window.timestamps),
        )

        return FatigueScore(
            level=level,
            confidence=confidence,
            perclos=perclos,
            blink_rate=blink_rate,
            yawn_count=yawn_count,
            head_down=head_down,
            face_absent=face_absent,
            events=events,
        )
    
    def _determine_level(
        self, perclos: float, yawn_count: int, head_down: bool, face_absent: bool
    ) -> FatigueLevel:
        """Determine fatigue level based on criteria."""
        severe_conditions = (
            perclos > self._config.PERCLOS_SEVERE,
            yawn_count >= 4,
            face_absent,
            perclos > self._config.PERCLOS_MILD and yawn_count >= 2,
            perclos > self._config.PERCLOS_MILD and head_down,
        )
        if any(severe_conditions):
            return "severe_fatigue"

        mild_conditions = (
            perclos > self._config.PERCLOS_MILD,
            yawn_count >= 2,
            head_down,
        )
        if any(mild_conditions):
            return "mild_fatigue"

        return "alert"
    
    def _calculate_confidence(
        self,
        level: FatigueLevel,
        perclos: float,
        yawn_count: int,
        head_down: bool,
        face_absent: bool,
        window_size: int,
    ) -> float:
        """Calculate confidence score."""
        # Short window
        if window_size < 10:
            return 0.5

        if level == "severe_fatigue":
            severe_conditions = (
                perclos > self._config.PERCLOS_SEVERE,
                yawn_count >= 4,
                face_absent,
                perclos > self._config.PERCLOS_MILD and yawn_count >= 2,
                perclos > self._config.PERCLOS_MILD and head_down,
            )
            criteria_met = sum(severe_conditions)
            
            return 1.0 if criteria_met >= 2 else 0.7

        if level == "mild_fatigue":
            return 0.6

        # Alert level
        # Check if window is long enough (>= PERCLOS_WINDOW_SEC)
        if window_size >= int(self._config.PERCLOS_WINDOW_SEC * 30):  # assuming 30fps
            return 0.9
        return 0.7