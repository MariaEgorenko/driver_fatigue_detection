import numpy as np
import pytest
import time

from ml.src.types import FaceLandmarks, FeatureWindow, YawnEvent
from ml.src.features.mouth_features import (
    MouthAnalyzer,
    MOUTH_TOP,
    MOUTH_BOTTOM,
    MOUTH_LEFT,
    MOUTH_RIGHT,
)


class MockConfig:
    """Mock configuration object."""
    MAR_THRESHOLD = 0.50
    YAWN_MIN_DURATION_SEC = 1.5


class TestMouthAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return MouthAnalyzer(MockConfig())
    
    def _create_landmarks(self, mouth_open: bool = False) -> FaceLandmarks:
        """Create synthetic landmarks with mouth configuration."""
        points = np.zeros((468, 3), dtype=np.float32)

        if mouth_open:
            # Open mouth: large vertical distance
            points[MOUTH_TOP] = [0.45, 0.5, 0.0]  # 13
            points[MOUTH_BOTTOM] = [0.45, 0.7, 0.0]  # 14
            points[MOUTH_LEFT] = [0.3, 0.6, 0.0]  # 78
            points[MOUTH_RIGHT] = [0.6, 0.6, 0.0]  # 308
        else:
            # Closed mouth: small vertical distance
            points[MOUTH_TOP] = [0.45, 0.55, 0.0]  # 13
            points[MOUTH_BOTTOM] = [0.45, 0.56, 0.0]  # 14
            points[MOUTH_LEFT] = [0.3, 0.55, 0.0]  # 78
            points[MOUTH_RIGHT] = [0.6, 0.55, 0.0]  # 308

        return FaceLandmarks(
            points=points, frame_width=640, frame_height=480, timestamp=time.time()
        )
    
    def test_mar_open_mouth(self, analyzer):
        """Test MAR with open mouth (height > width)."""
        landmarks = self._create_landmarks(mouth_open=True)
        mar = analyzer.compute_mar(landmarks)

        # MAR = 0.2/0.3 ≈ 0.667 > 0.5
        assert mar > 0.5

    def test_mar_closed_mouth(self, analyzer):
        """Test MAR with closed mouth (height ≈ 0)."""
        landmarks = self._create_landmarks(mouth_open=False)
        mar = analyzer.compute_mar(landmarks)

        # MAR ≈ 0.033 ≈ 0.0
        assert abs(mar - 0.0) < 0.1

    def test_detect_yawn_complete(self, analyzer):
        """Test detection of complete yawn (frame by frame)."""
        window = FeatureWindow()
        yawns = []
        
        mars = [0.2, 0.8, 0.9, 0.85, 0.9, 0.2, 0.2]
        timestamps = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

        # Simulate streaming processing of frames
        for m, t in zip(mars, timestamps):
            window.mar_history.append(m)
            window.timestamps.append(t)
            yawns.extend(analyzer.detect_yawn(window))

        assert len(yawns) == 1
        
        assert yawns[0].duration_sec >= 1.5
        assert yawns[0].max_mar == pytest.approx(0.9, abs=0.01)

    def test_detect_yawn_too_short(self, analyzer):
        """Test that short MAR spike is not detected as yawn."""
        window = FeatureWindow()
        yawns = []
        
        mars = [0.2, 0.8, 0.2, 0.2, 0.2, 0.2]
        timestamps = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

        for m, t in zip(mars, timestamps):
            window.mar_history.append(m)
            window.timestamps.append(t)
            yawns.extend(analyzer.detect_yawn(window))

        assert len(yawns) == 0

    def test_detect_yawn_ongoing(self, analyzer):
        """Test that ongoing yawn (not finished) returns None (no events)."""
        window = FeatureWindow()
        yawns = []
        
        mars = [0.2, 0.8, 0.9, 0.85]
        timestamps = [1.0, 2.0, 3.0, 4.0]

        for m, t in zip(mars, timestamps):
            window.mar_history.append(m)
            window.timestamps.append(t)
            yawns.extend(analyzer.detect_yawn(window))

        assert len(yawns) == 0
