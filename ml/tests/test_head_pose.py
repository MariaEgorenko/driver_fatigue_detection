import numpy as np
import cv2
import pytest
import time

from ml.src.types import FaceLandmarks, HeadPose
from ml.src.features.head_pose import HeadPoseEstimator, MODEL_POINTS_3D


class TestHeadPoseEstimator:
    @pytest.fixture
    def estimator(self):
        return HeadPoseEstimator()
    
    def _create_neutral_landmarks(self, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0) -> FaceLandmarks:
        """Creates 2D points by perfectly projecting a 3D model (cv2.projectPoints)."""
        frame_width = 640
        frame_height = 480
        
        focal_length = float(frame_width)
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        rvec = np.array([
            np.deg2rad(pitch_deg), 
            np.deg2rad(yaw_deg), 
            np.deg2rad(roll_deg)
        ], dtype=np.float64)
        
        tvec = np.array([0.0, 0.0, 1500.0], dtype=np.float64)

        image_points, _ = cv2.projectPoints(
            MODEL_POINTS_3D, rvec, tvec, camera_matrix, dist_coeffs
        )
        
        image_points = image_points.squeeze()

        points_norm = np.zeros((468, 3), dtype=np.float32)
        
        indices = [1, 152, 263, 33, 287, 57]
        for i, idx in enumerate(indices):
            points_norm[idx][0] = image_points[i][0] / frame_width
            points_norm[idx][1] = image_points[i][1] / frame_height
            points_norm[idx][2] = 0.0 

        return FaceLandmarks(
            points=points_norm, 
            frame_width=frame_width, 
            frame_height=frame_height, 
            timestamp=time.time()
        )
    
    def test_estimate_returns_headpose_type(self, estimator):
        """Test that estimate returns HeadPose object with perfect 0 angles."""
        landmarks = self._create_neutral_landmarks(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0)
        result = estimator.estimate(landmarks)

        assert result is not None, "solvePnP failed to converge"
        assert isinstance(result, HeadPose)
        
        assert abs(result.pitch) < 1e-3
        assert abs(result.yaw) < 1e-3
        assert abs(result.roll) < 1e-3

    def test_is_head_down_true(self, estimator):
        """Test is_head_down returns True for large pitch."""
        for _ in range(30):
            estimator.estimate(self._create_neutral_landmarks(pitch_deg=0.0))

        assert estimator._baseline_pitch is not None
        assert abs(estimator._baseline_pitch) < 1e-3

        pose = HeadPose(pitch=25.0, yaw=0.0, roll=0.0)
        result = estimator.is_head_down(pose, threshold_deg=20.0)
        assert result is True

    def test_is_head_down_false(self, estimator):
        """Test is_head_down returns False for small pitch."""
        pose = HeadPose(pitch=10.0, yaw=0.0, roll=0.0)
        result = estimator.is_head_down(pose, threshold_deg=20.0)
        assert result is False

    def test_is_head_down_high_roll_override(self, estimator):
        """Test that extreme roll angle disables head down detection."""
        pose = HeadPose(pitch=50.0, yaw=0.0, roll=50.0)  # roll > 45
        result = estimator.is_head_down(pose, threshold_deg=20.0)
        assert result is False

    def test_camera_matrix_auto_init(self, estimator):
        """Test that camera_matrix is auto-initialized when not set."""
        assert estimator._camera_matrix is None

        landmarks = self._create_neutral_landmarks()
        result = estimator.estimate(landmarks)

        assert estimator._camera_matrix is not None
        assert estimator._camera_matrix.shape == (3, 3)
        assert result is not None
