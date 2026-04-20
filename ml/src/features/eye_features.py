import numpy as np
from typing import List, Optional

from ml.src.types import (
    FaceLandmarks, BlinkEvent,
    FeatureWindow, FatigueEvent
)
from ml.src.features.utils import euclidean


# MediaPipe indexes for the right eye (relative to the entire 468-point grid)
# Order: p1(outer corner), p2(top), p3(top), p4(inner corner), p5(bottom), p6(bottom)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

class EyeAnalyzer:
    def __init__(self, config) -> None:
        self._ear_threshold = config.EAR_THRESHOLD
        self._blink_max_ms = config.BLINK_MAX_DURATION_MS
        self._blink_min_ms = config.BLINK_MIN_DURATION_MS
        self._perclos_window = config.PERCLOS_WINDOW_SEC

        # Blink detection state
        self._in_closure = False
        self._closure_start_time: Optional[float] = None
        self._min_ear_in_closure = 1.0

    def compute_ear(self, landmarks: FaceLandmarks) -> float:
        """The average EAR for both eyes according to the Soukupova & Cech formula."""
        points = landmarks.points
        w = landmarks.frame_width
        h = landmarks.frame_height

        def to_px(pt):
            """Normalized point (x,y,z) -> pixel (x,y)"""
            return np.array([pt[0] * w, pt[1] * h])

        # Right eye
        r_p1, r_p2, r_p3, r_p4, r_p5, r_p6 = [to_px(points[i]) for i in RIGHT_EYE_INDICES]
        r_ear = (euclidean(r_p2, r_p6) + euclidean(r_p3, r_p5)) / (2 * euclidean(r_p1, r_p4) + 1e-6)

        # Left eye
        l_p1, l_p2, l_p3, l_p4, l_p5, l_p6 = [to_px(points[i]) for i in LEFT_EYE_INDICES]
        l_ear = (euclidean(l_p2, l_p6) + euclidean(l_p3, l_p5)) / (2 * euclidean(l_p1, l_p4) + 1e-6)

        return (r_ear + l_ear) / 2.0
    
    def compute_perclos(self, window: FeatureWindow) -> float:
        """PERCLOS = the percentage of frames per window,where EAR < threshold."""
        if len(window.ear_history) < 2 or not window.timestamps:
            return 0.0
        
        current_time = window.timestamps[-1]
        window_start = current_time - self._perclos_window

        # Filtering the values within the window
        valid_ears = [
            ear for ear, ts in zip(window.ear_history, window.timestamps)
            if ts >= window_start and ear > 0.0
        ]

        if len(valid_ears) < 30:
            return 0.0
        
        closed_count = sum(1 for ear in valid_ears if ear < self._ear_threshold)
        return closed_count / len(valid_ears)
    
    def detect_blinks_and_closures(self, window: FeatureWindow) -> List[BlinkEvent]:
        """Return the list of new blinkevents for the last added frame.
        
        State machine:
        - open: the eye is open (EAR >= threshold)
        - closing: the eye has started to close (EAR < threshold), we are waiting for the opening
        - closed: the eye is closed, we check the duration when opening
        
        Blinking is counted only if the closed -> open transition is successful.,
        if the duration is <= BLINK_MAX_DURATION_MS.
        """
        if not window.ear_history or not window.timestamps:
            return []
        
        last_ear = window.ear_history[-1]
        last_timestamp = window.timestamps[-1]

        # Skip placeholders
        if last_ear <= 0.0:
            return []
        
        blinks = []
        closures = []

        if not self._in_closure:
            # Currently open, check if started closing
            if last_ear < self._ear_threshold:
                self._in_closure = True
                self._closure_start_time = last_timestamp
                self._min_ear_in_closure = last_ear
        else:
            # Currently closing/closed, check if opened again
            if last_ear >= self._ear_threshold:
                if self._closure_start_time is not None:
                    duration_sec = last_timestamp - self._closure_start_time
                    duration_ms = duration_sec * 1000

                    if self._blink_min_ms <= duration_ms <= self._blink_max_ms:
                        blinks.append(
                            BlinkEvent(
                                timestamp=last_timestamp,
                                duration_ms=duration_ms,
                                min_ear=self._min_ear_in_closure,
                            )
                        )
                    elif duration_ms > self._blink_max_ms:
                        severity = "high" if duration_sec >= 1.5 else "medium"
                        closures.append(
                            FatigueEvent(
                                event_type="eye_closure",
                                timestamp=self._closure_start_time,
                                duration_sec=duration_sec,
                                severity=severity,
                                metadata={"min_ear": self._min_ear_in_closure}
                            )
                        )

                # Reset state
                self._in_closure = False
                self._closure_start_time = None
                self._min_ear_in_closure = 1.0
            else:
                # Continue closing
                self._min_ear_in_closure = min(self._min_ear_in_closure, last_ear)

        return blinks, closures
    
    def blink_rate(self, window: FeatureWindow) -> float:
        """Blink rate in blinks per minute.
        
        We store blinks for the last 60 seconds.
        blink_rate = number of blinks / time_of the first_organization in_minutes.
        """
        if not window.blink_events or len(window.timestamps) < 2:
            return 0.0
        
        current_time = window.timestamps[-1]
        window_start_time = window.timestamps[0]

        time_span_minutes = (current_time - window_start_time) / 60.0

        # At least 10 seconds of data to calculate a meaningful frequency
        if time_span_minutes < 0.16:
            return 0.0
        
        blink_count = len(window.blink_events)
        return blink_count / time_span_minutes