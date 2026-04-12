import pytest
import time

from ml.src.types import FeatureWindow, YawnEvent, AbsenceEvent, HeadPose
from ml.src.inference.fatigue_scorer import FatigueScorer


class MockConfig:
    """Mock configuration object."""
    EAR_THRESHOLD = 0.20
    MAR_THRESHOLD = 0.50
    YAWN_MIN_DURATION_SEC = 1.5
    HEAD_PITCH_THRESHOLD_DEG = 20.0
    FACE_ABSENT_THRESHOLD_SEC = 3.0
    PERCLOS_WINDOW_SEC = 60.0
    PERCLOS_MILD = 0.35
    PERCLOS_SEVERE = 0.70
    BLINK_MAX_DURATION_MS = 400
    BLINK_MIN_DURATION_MS = 50


class TestFatigueScorer:
    @pytest.fixture
    def scorer(self):
        return FatigueScorer(MockConfig())
    
    def test_empty_window(self, scorer):
        """Test empty window returns alert with confidence 0.0."""
        window = FeatureWindow()

        score = scorer.score(window)

        assert score.level == "alert"
        assert score.confidence == 0.0
        assert score.perclos == 0.0
        assert score.blink_rate == 0.0
        assert score.yawn_count == 0
        assert score.head_down is False
        assert score.face_absent is False
        assert len(score.events) == 0

    def test_severe_high_perclos(self, scorer):
        """Test PERCLOS=0.8 returns severe_fatigue."""
        current_time = time.time()
        n_frames = 15
        
        timestamps = [current_time - n_frames + i for i in range(n_frames)]

        window = FeatureWindow(
            ear_history=[0.05] * n_frames,  # All closed (< 0.20)
            timestamps=timestamps,
            mar_history=[0.1] * n_frames,
            head_poses=[HeadPose(pitch=0, yaw=0, roll=0)] * n_frames,
            blink_events=[],
            yawn_events=[],
            absence_events=[],
            face_detected_flags=[True] * n_frames,
        )

        score = scorer.score(window)

        assert score.level == "severe_fatigue"
        assert score.perclos > 0.70
        assert score.confidence >= 0.7

    def test_severe_many_yawns(self, scorer):
        """Test 4+ yawns in 5 min returns severe_fatigue."""
        current_time = time.time()

        yawns = [
            YawnEvent(timestamp=current_time - 10, duration_sec=2.0, max_mar=0.8),
            YawnEvent(timestamp=current_time - 60, duration_sec=2.0, max_mar=0.8),
            YawnEvent(timestamp=current_time - 180, duration_sec=2.0, max_mar=0.8),
            YawnEvent(timestamp=current_time - 280, duration_sec=2.0, max_mar=0.8),
        ]

        window = FeatureWindow(
            ear_history=[0.5] * 5,
            timestamps=[current_time - 5 + i for i in range(5)],
            mar_history=[0.1] * 5,
            head_poses=[HeadPose(pitch=0, yaw=0, roll=0)] * 5,
            blink_events=[],
            yawn_events=yawns,
            absence_events=[],
            face_detected_flags=[True] * 5,
        )

        score = scorer.score(window)

        assert score.level == "severe_fatigue"
        assert score.yawn_count == 4

    def test_mild_medium_perclos(self, scorer):
        """Test PERCLOS=0.40 returns mild_fatigue."""
        current_time = time.time()
        n_frames = 10
        
        timestamps = [current_time - n_frames + i for i in range(n_frames)]
        ear_history = [0.5] * 6 + [0.1] * 4  # 4 closed out of 10 = 0.40

        window = FeatureWindow(
            ear_history=ear_history,
            timestamps=timestamps,
            mar_history=[0.1] * n_frames,
            head_poses=[HeadPose(pitch=0, yaw=0, roll=0)] * n_frames,
            blink_events=[],
            yawn_events=[],
            absence_events=[],
            face_detected_flags=[True] * n_frames,
        )

        score = scorer.score(window)

        assert score.level == "mild_fatigue"
        assert abs(score.perclos - 0.4) < 0.1
        assert score.confidence == 0.6

    def test_mild_two_yawns(self, scorer):
        """Test 2 yawns returns mild_fatigue."""
        current_time = time.time()

        yawns = [
            YawnEvent(timestamp=current_time - 10, duration_sec=2.0, max_mar=0.8),
            YawnEvent(timestamp=current_time - 100, duration_sec=2.0, max_mar=0.8),
        ]

        window = FeatureWindow(
            ear_history=[0.5] * 5,
            timestamps=[current_time - 5 + i for i in range(5)],
            mar_history=[0.1] * 5,
            head_poses=[HeadPose(pitch=0, yaw=0, roll=0)] * 5,
            blink_events=[],
            yawn_events=yawns,
            absence_events=[],
            face_detected_flags=[True] * 5,
        )

        score = scorer.score(window)

        assert score.level == "mild_fatigue"
        assert score.yawn_count == 2

    def test_alert_normal(self, scorer):
        """Test normal metrics return alert."""
        current_time = time.time()

        window = FeatureWindow(
            ear_history=[0.5] * 5,
            timestamps=[current_time - 5 + i for i in range(5)],
            mar_history=[0.1] * 5,
            head_poses=[HeadPose(pitch=0, yaw=0, roll=0)] * 5,
            blink_events=[],
            yawn_events=[],
            absence_events=[],
            face_detected_flags=[True] * 5,
        )

        score = scorer.score(window)

        assert score.level == "alert"
        assert score.perclos == 0.0
        assert score.yawn_count == 0
        assert not score.head_down
        assert not score.face_absent

    def test_combined_severe(self, scorer):
        """Test combined criteria: PERCLOS > mild + yawn >= 2 → severe."""
        current_time = time.time()

        yawns = [
            YawnEvent(timestamp=current_time - 10, duration_sec=2.0, max_mar=0.8),
            YawnEvent(timestamp=current_time - 100, duration_sec=2.0, max_mar=0.8),
        ]

        window = FeatureWindow(
            ear_history=[0.5, 0.5, 0.1, 0.1, 0.5],  # 0.40 PERCLOS
            timestamps=[current_time - 5 + i for i in range(5)],
            mar_history=[0.1] * 5,
            head_poses=[HeadPose(pitch=0, yaw=0, roll=0)] * 5,
            blink_events=[],
            yawn_events=yawns,
            absence_events=[],
            face_detected_flags=[True] * 5,
        )

        score = scorer.score(window)

        assert score.level == "severe_fatigue"
        assert score.perclos > 0.35
        assert score.yawn_count == 2

    def test_mild_head_down(self, scorer):
        """Test that 3+ frames of head down triggers mild_fatigue."""
        current_time = time.time()
        n_frames = 10
        
        # 7 normal frames, 3 frames with pitch=25.0 (threshold is 20)
        poses = [HeadPose(pitch=0, yaw=0, roll=0)] * 7 + [HeadPose(pitch=25.0, yaw=0, roll=0)] * 3

        window = FeatureWindow(
            ear_history=[0.5] * n_frames,
            timestamps=[current_time - n_frames + i for i in range(n_frames)],
            mar_history=[0.1] * n_frames,
            head_poses=poses,
            blink_events=[],
            yawn_events=[],
            absence_events=[],
            face_detected_flags=[True] * n_frames,
        )

        score = scorer.score(window)

        assert score.head_down is True
        assert score.level == "mild_fatigue"

    def test_severe_face_absent(self, scorer):
        """Test that missing face triggers severe_fatigue."""
        current_time = time.time()
        n_frames = 5
        
        # The face disappeared in the last frame (face_detected_flags[-1] == False)
        # And there is an event confirming that it has been missing for longer than the threshold.
        absence_events = [
            AbsenceEvent(start_time=current_time - 4.0, duration_sec=4.0)
        ]

        window = FeatureWindow(
            ear_history=[0.5] * n_frames,
            timestamps=[current_time - n_frames + i for i in range(n_frames)],
            mar_history=[0.1] * n_frames,
            head_poses=[HeadPose(pitch=0, yaw=0, roll=0)] * n_frames,
            blink_events=[],
            yawn_events=[],
            absence_events=absence_events,
            face_detected_flags=[True, True, True, True, False],
        )

        score = scorer.score(window)

        assert score.face_absent is True
        assert score.level == "severe_fatigue"
