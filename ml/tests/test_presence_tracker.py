import pytest

from ml.src.types import AbsenceEvent
from ml.src.features.presence_tracker import FacePresenceTracker


class TestFacePresenceTracker:
    @pytest.fixture
    def tracker(self):
        return FacePresenceTracker(threshold_sec=3.0)

    def test_no_event_below_threshold(self, tracker):
        """Test no event when absence is below threshold."""
        # Simulate face absent for 2 seconds (threshold is 3)
        tracker.update(face_detected=False, timestamp=0.0)
        tracker.update(face_detected=False, timestamp=1.0)
        tracker.update(face_detected=False, timestamp=2.0)

        # Should not return event yet (only 2 seconds)
        event = tracker.update(face_detected=False, timestamp=2.5)
        assert event is None

    def test_event_above_threshold(self, tracker):
        """Test event returned when absence exceeds threshold."""
        # Start absence
        tracker.update(face_detected=False, timestamp=0.0)

        # Check at 2.9s - still below threshold
        event = tracker.update(face_detected=False, timestamp=2.9)
        assert event is None

        # Check at 4.0s - above threshold (3.0s)
        event = tracker.update(face_detected=False, timestamp=4.0)
        assert event is not None
        assert isinstance(event, AbsenceEvent)
        assert event.duration_sec == pytest.approx(4.0, abs=0.1)
        assert event.start_time == 0.0

    def test_single_event_per_episode(self, tracker):
        """Test that only one event is returned per absence episode."""
        # Start absence
        tracker.update(face_detected=False, timestamp=0.0)

        # First event at 4.0s
        event1 = tracker.update(face_detected=False, timestamp=4.0)
        assert event1 is not None

        # Second check at 6.0s - should NOT return new event (same episode)
        event2 = tracker.update(face_detected=False, timestamp=6.0)
        assert event2 is None

        # Third check at 10.0s - still no event
        event3 = tracker.update(face_detected=False, timestamp=10.0)
        assert event3 is None

    def test_reset_on_detection(self, tracker):
        """Test that face detection resets the tracker."""
        # First absence episode
        tracker.update(face_detected=False, timestamp=0.0)
        event1 = tracker.update(face_detected=False, timestamp=4.0)
        assert event1 is not None

        # Face detected - should reset
        tracker.update(face_detected=True, timestamp=5.0)

        # New absence episode
        tracker.update(face_detected=False, timestamp=6.0)

        # Should trigger new event at 10.0s (4 seconds absence)
        event2 = tracker.update(face_detected=False, timestamp=10.0)
        assert event2 is not None
        assert event2.start_time == 6.0

    def test_reset_method(self, tracker):
        """Test that reset() clears the state."""
        # Start absence
        tracker.update(face_detected=False, timestamp=0.0)

        # Call reset
        tracker.reset()

        # Now simulate new absence - should start fresh
        tracker.update(face_detected=False, timestamp=10.0)

        # At 12.0s - only 2 seconds, below threshold
        event = tracker.update(face_detected=False, timestamp=12.0)
        assert event is None

        # At 14.0s - 4 seconds, above threshold
        event = tracker.update(face_detected=False, timestamp=14.0)
        assert event is not None
        assert event.start_time == 10.0

    def test_continuous_presence(self, tracker):
        """Test that continuous presence doesn't cause any errors or events."""
        event1 = tracker.update(face_detected=True, timestamp=0.0)
        event2 = tracker.update(face_detected=True, timestamp=2.0)
        event3 = tracker.update(face_detected=True, timestamp=5.0)

        assert event1 is None
        assert event2 is None
        assert event3 is None
        
        assert tracker._absence_start is None
