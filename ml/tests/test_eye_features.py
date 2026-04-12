import numpy as np
import pytest
import time

from ml.src.types import FaceLandmarks, FeatureWindow, BlinkEvent
from ml.src.features.eye_features import EyeAnalyzer


class MockConfig:
    """Mock configuration object."""
    EAR_THRESHOLD = 0.20
    BLINK_MAX_DURATION_MS = 400
    BLINK_MIN_DURATION_MS = 50
    PERCLOS_WINDOW_SEC = 60.0


class TestEyeAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return EyeAnalyzer(MockConfig())
    
    def _create_landmarks(self, ear_value: float = 0.5) -> FaceLandmarks:
        """Create synthetic landmarks with specific EAR value."""
        points = np.zeros((468, 3), dtype=np.float32)

        # Right eye: set width=2.0, height=ear_value*2.0
        points[33] = [0.3, 0.4, 0.0]  # p1 outer
        points[160] = [0.35, 0.35, 0.0]  # p2 top
        points[158] = [0.45, 0.35, 0.0]  # p3 top
        points[133] = [0.5, 0.4, 0.0]  # p4 inner
        points[153] = [0.45, 0.45, 0.0]  # p5 bottom
        points[144] = [0.35, 0.45, 0.0]  # p6 bottom

        # Left eye indices
        points[362] = [0.6, 0.4, 0.0]  # p1 outer
        points[385] = [0.65, 0.35, 0.0]  # p2 top
        points[387] = [0.75, 0.35, 0.0]  # p3 top
        points[263] = [0.8, 0.4, 0.0]  # p4 inner
        points[373] = [0.75, 0.45, 0.0]  # p5 bottom
        points[380] = [0.65, 0.45, 0.0]  # p6 bottom

        return FaceLandmarks(
            points=points, frame_width=640, frame_height=480, timestamp=time.time()
        )
    
    def test_ear_known_value(self, analyzer):
        """Test EAR calculation with known geometry."""
        landmarks = self._create_landmarks(ear_value=1.0)
        ear = analyzer.compute_ear(landmarks)
        assert 0.4 < ear < 0.6  # Expected ~0.5 for the geometry

    def test_ear_closed_eye(self, analyzer):
        """Test EAR with closed eye (zero height)."""
        points = np.zeros((468, 3), dtype=np.float32)

        # Right eye closed (all points on same horizontal line Y=0.4)
        points[33] = [0.3, 0.4, 0.0]
        points[160] = [0.35, 0.4, 0.0]
        points[158] = [0.45, 0.4, 0.0]
        points[133] = [0.5, 0.4, 0.0]
        points[153] = [0.45, 0.4, 0.0]
        points[144] = [0.35, 0.4, 0.0]

        # Left eye closed
        points[362] = [0.6, 0.4, 0.0]
        points[385] = [0.65, 0.4, 0.0]
        points[387] = [0.75, 0.4, 0.0]
        points[263] = [0.8, 0.4, 0.0]
        points[373] = [0.75, 0.4, 0.0]
        points[380] = [0.65, 0.4, 0.0]

        landmarks = FaceLandmarks(
            points=points, frame_width=640, frame_height=480, timestamp=time.time()
        )

        ear = analyzer.compute_ear(landmarks)
        assert abs(ear - 0.0) < 0.01

    def test_perclos_all_closed(self, analyzer):
        """Test PERCLOS with all eyes closed."""
        window = FeatureWindow(
            ear_history=[0.1, 0.1, 0.1, 0.1, 0.1], timestamps=[1.0, 2.0, 3.0, 4.0, 5.0]
        )

        perclos = analyzer.compute_perclos(window)
        assert perclos == 1.0

    def test_perclos_all_open(self, analyzer):
        """Test PERCLOS with all eyes open."""
        window = FeatureWindow(
            ear_history=[0.5, 0.5, 0.5, 0.5, 0.5], timestamps=[1.0, 2.0, 3.0, 4.0, 5.0]
        )

        perclos = analyzer.compute_perclos(window)
        assert perclos == 0.0

    def test_perclos_empty_window(self, analyzer):
        """Test PERCLOS with empty window."""
        window = FeatureWindow(ear_history=[], timestamps=[])

        perclos = analyzer.compute_perclos(window)
        assert perclos == 0.0

    def test_detect_blinks_single_blink(self, analyzer):
        """Test detection of single blink by simulating sequential frames."""
        ears = [0.5, 0.5, 0.1, 0.1, 0.5, 0.5]
        timestamps = [1.0, 1.1, 1.2, 1.4, 1.5, 1.6]

        detected_blinks = []
        window = FeatureWindow()
        
        # We simulate the frame processing cycle by feeding them one at a time
        for ear, ts in zip(ears, timestamps):
            window.ear_history.append(ear)
            window.timestamps.append(ts)
            detected_blinks.extend(analyzer.detect_blinks(window))

        assert len(detected_blinks) == 1
        # Blinking started at 1.2 (EAR=0.1) and ended at 1.5 (EAR=0.5)
        # Duration: (1.5 - 1.2) = 0.3s = 300ms
        assert abs(detected_blinks[0].duration_ms - 300) < 1e-5
        assert detected_blinks[0].min_ear == 0.1

    def test_detect_blinks_no_blink_long_closure(self, analyzer):
        """Test that long closure (>400ms) is not detected as blink."""
        ears = [0.5, 0.5, 0.1, 0.1, 0.1, 0.5]
        timestamps = [1.0, 1.1, 1.2, 1.5, 1.8, 2.0]  # closure from 1.2 to 2.0 = 800ms

        detected_blinks = []
        window = FeatureWindow()
        
        # Simulating the processing cycle
        for ear, ts in zip(ears, timestamps):
            window.ear_history.append(ear)
            window.timestamps.append(ts)
            detected_blinks.extend(analyzer.detect_blinks(window))

        assert len(detected_blinks) == 0  # Not blinking, because 800ms > 400ms

    def test_blink_rate(self, analyzer):
        """Test blink rate calculation."""
        window = FeatureWindow(
            ear_history=[0.5] * 60, 
            timestamps=[float(i) for i in range(60)]
        )
        window.blink_events = [
            BlinkEvent(timestamp=10.0, duration_ms=200, min_ear=0.1),
            BlinkEvent(timestamp=30.0, duration_ms=200, min_ear=0.1),
            BlinkEvent(timestamp=50.0, duration_ms=200, min_ear=0.1),
        ]

        blink_rate = analyzer.blink_rate(window)
        # 3 blinks / (59 / 60) minutes ≈ 3.05 bpm
        assert abs(blink_rate - 3.05) < 0.2