import numpy as np
from typing import List, Optional

from ml.src.types import FaceLandmarks, YawnEvent, FeatureWindow
from ml.src.features.utils import euclidean

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


class MouthAnalyzer:
    def __init__(self, config) -> None:
        self._mar_threshold = config.MAR_THRESHOLD
        self._yawn_min_duration = config.YAWN_MIN_DURATION_SEC
        
        # State machine
        self._yawn_state = "closed"  # States: closed, opening, yawning
        self._yawn_start_time: Optional[float] = None
        self._max_mar_in_yawn = 0.0

        # Hysteresis for "rattle"
        self._hysteresis_threshold = self._mar_threshold * 0.8
        self._false_start_counter = 0
        self._false_start_max = 3  # Allow 3 consecutive low frames before reset

        # Minimum time mouth must stay closed to complete yawn
        self._min_closed_duration = 0.3
        self._closed_start_time: Optional[float] = None

        self._last_processed_timestamp = 0.0

    def compute_mar(self, landmarks: FaceLandmarks) -> float:
        """Mouth Aspect Ratio.
        MAR = ||top - bottom|| / ||left - right||
        """
        points = landmarks.points

        top = points[MOUTH_TOP]
        bottom = points[MOUTH_BOTTOM]
        left = points[MOUTH_LEFT]
        right = points[MOUTH_RIGHT]

        vertical = euclidean(top, bottom)
        horizontal = euclidean(left, right)

        if horizontal == 0:
            return 0.0

        return vertical / horizontal
    
    def detect_yawn(self, window: FeatureWindow) -> List[YawnEvent]:
        """Yawn detection via the state machine based.
        
        Analyzes ALL new frames since the last call.
        Uses hysteresis to protect against rattling.
        
        States:
        - closed: mouth is closed (MAR < threshold)
        - opening: mouth has started to open (MAR > threshold), waiting for confirmation
        - yawning: mouth open >= YAWN_MIN_DURATION_SEC, waiting for closure
        
        Returns:
        List[YawnEvent] - can contain 0 or more events
        """
        yawns: List[YawnEvent] = []
        
        if not window.mar_history or not window.timestamps:
            return yawns
        
        valid_new_data = [
            (mar, ts) for mar, ts in zip(window.mar_history, window.timestamps)
            if mar > 0.0 and ts > self._last_processed_timestamp
        ]
        
        if not valid_new_data:
            return yawns
        
        # Process all new frames since last call
        for mar, timestamp in valid_new_data:
            self._process_single_yawn(mar, timestamp, yawns)
            
            self._last_processed_timestamp = timestamp
            
        return yawns
    
    def _process_single_yawn(
        self, mar: float, timestamp: float, yawns: List[YawnEvent]
    ) -> None:
        """Process single MAR value for state machine."""
        
        if self._yawn_state == "closed":
            if mar > self._mar_threshold:
                self._yawn_state = "opening"
                self._yawn_start_time = timestamp
                self._max_mar_in_yawn = mar
                self._false_start_counter = 0
                self._closed_start_time = None
                
        elif self._yawn_state == "opening":
            if mar > self._mar_threshold:
                # Continue opening
                self._max_mar_in_yawn = max(self._max_mar_in_yawn, mar)
                
                if self._yawn_start_time is not None:
                    duration = timestamp - self._yawn_start_time
                    if duration >= self._yawn_min_duration:
                        # Confirmed as yawn
                        self._yawn_state = "yawning"
                        self._false_start_counter = 0
                        
            elif mar < self._hysteresis_threshold:
                # Use lower threshold to prevent false resets
                self._false_start_counter += 1
                if self._false_start_counter >= self._false_start_max:
                    self._yawn_state = "closed"
                    self._yawn_start_time = None
                    self._max_mar_in_yawn = 0.0
                    self._false_start_counter = 0
                    
        elif self._yawn_state == "yawning":
            if mar <= self._mar_threshold:
                if self._closed_start_time is None:
                    self._closed_start_time = timestamp
                elif timestamp - self._closed_start_time >= self._min_closed_duration:
                    self._yawn_state = "closed"
                    
                    if self._yawn_start_time is not None:
                        yawns.append(
                            YawnEvent(
                                timestamp=timestamp,
                                duration_sec=timestamp - self._yawn_start_time,
                                max_mar=self._max_mar_in_yawn
                            )
                        )
                        
                    # Reset parameters
                    self._yawn_start_time = None
                    self._max_mar_in_yawn = 0.0
                    self._closed_start_time = None
            else:
                self._max_mar_in_yawn = max(self._max_mar_in_yawn, mar)
                self._closed_start_time = None
