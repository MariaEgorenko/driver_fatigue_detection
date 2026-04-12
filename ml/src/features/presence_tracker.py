from typing import Optional
import time

from ml.src.types import AbsenceEvent


class FacePresenceTracker:
    """Monitors the continuous absence of a face in the frame."""

    def __init__(self, threshold_sec: float) -> None:
        self._threshold = threshold_sec
        self._absence_start: Optional[float] = None
        
        self._event_reported: bool = False

    def update(
        self,
        face_detected: bool,
        timestamp: float,
    ) -> Optional[AbsenceEvent]:
        """Call for each frame.

        Returns AbsenceEvent when:
        1. The person is missing continuously > threshold_sec
        2. The event has not been returned for this episode yet.
           (one episode = one event, even if it lasts 10 seconds)

        Resets the countdown when the face reappears.
        """
        if face_detected:
            # Face detected - reset absence tracking
            self._absence_start = None
            self._event_reported = False
            return None

        # Face not detected
        if self._absence_start is None:
            # Start tracking absence
            self._absence_start = timestamp
            return None

        # Calculate duration of absence
        duration = timestamp - self._absence_start

        if duration > self._threshold:
            # Check if we already reported an event for this episode
            if not self._event_reported:
                # Report new event
                event = AbsenceEvent(
                    start_time=self._absence_start, 
                    duration_sec=duration
                )
                self._event_reported = True
                return event

        return None
    
    def reset(self) -> None:
        """Reset the tracker status."""
        self._absence_start = None
        self._event_reported = False
