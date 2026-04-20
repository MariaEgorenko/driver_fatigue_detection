import numpy as np
import cv2
from typing import Optional

from ml.src.types import FaceLandmarks, HeadPose

# Adaptation of the 3D model to the axes of the OpenCV camera
# OpenCV: X - to the right, Y - down, Z - away from the camera.
# Therefore, the chin (at the bottom) has a +Y, and the eyes (at the top) have a -Y.
# The right side of the face (in the picture on the left) has -X, the left (in the picture on the right) has +X.
MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # 1: Tip of the nose (center)
        [0.0, 330.0, -65.0],      # 152: Chin (shifted down the Y)
        [225.0, -170.0, 135.0],  # 263: Left corner of the eye (shifted up by -Y, right by +X)
        [-225.0, -170.0, 135.0], # 33: Right corner of the eye (shifted up by Y, left by X)
        [150.0, 150.0, 125.0],   # 287: The left corner of the mouth
        [-150.0, 150.0, -125.0],  # 57: Right corner of the mouth
    ],
    dtype=np.float64,
)

LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]


class HeadPoseEstimator:
    def __init__(self) -> None:
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._last_frame_size = (0, 0)

        self._baseline_pitch: Optional[float] = None
        self._calibration_pitches: list[float] = []
        self._calibration_frames = 30

    def reset_calibration(self) -> None:
        """Reset calibration when starting a new session."""
        self._baseline_pitch = None
        self._calibration_pitches.clear()

    def set_camera_matrix(
        self,
        focal_length: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Install the camera matrix."""
        center = (frame_width / 2, frame_height / 2)
        self._camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self._last_frame_size = (frame_width, frame_height)

    def estimate(
        self,
        landmarks: FaceLandmarks,
    ) -> Optional[HeadPose]:
        """Assessment of head position (Pitch, Yaw, Roll)."""
        
        current_size = (landmarks.frame_width, landmarks.frame_height)
        if self._camera_matrix is None or self._last_frame_size != current_size:
            self.set_camera_matrix(
                focal_length=float(landmarks.frame_width),
                frame_width=landmarks.frame_width,
                frame_height=landmarks.frame_height,
            )

        points_2d = landmarks.points[LANDMARK_INDICES, :2].astype(np.float64).copy()

        # Convert normalized coordinates [0,1] to pixel coordinates
        points_2d[:, 0] *= landmarks.frame_width
        points_2d[:, 1] *= landmarks.frame_height

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            points_2d,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
        
        pitch = euler_angles[0]
        yaw = euler_angles[1]
        roll = euler_angles[2]

        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
            
        pose = HeadPose(pitch=pitch, yaw=yaw, roll=roll)

        if self._baseline_pitch is None:
            self._calibration_pitches.append(pose.pitch)
            if len(self._calibration_pitches) >= self._calibration_frames:
                self._baseline_pitch = float(np.mean(self._calibration_pitches))

        return pose
    
    def is_head_down(
        self,
        pose: HeadPose,
        threshold_deg: float,
    ) -> bool:
        """pitch > threshold_deg → The head is lowered."""
        if abs(pose.roll) > 45:
            return False
        
        if self._baseline_pitch is not None:
            deviation = pose.pitch - self._baseline_pitch
            return deviation > threshold_deg
        
        return pose.pitch > (threshold_deg * 1.5)