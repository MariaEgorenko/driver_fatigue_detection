import cv2
import numpy as np
import time
from typing import Optional

import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

from ml.src.types import FaceLandmarks


class FaceDetector:
    """Face detector via MediaPipe Face Mesh."""

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        is_video_stream: bool = True,
    ) -> None:
        self.max_num_faces = max_num_faces
        self.is_video_stream = is_video_stream

        running_mode = (
            VisionTaskRunningMode.VIDEO if is_video_stream else VisionTaskRunningMode.IMAGE
        )

        base_options = BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._detector = FaceLandmarker.create_from_options(options)
        self._closed = False

    def detect(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Optional[FaceLandmarks]:
        """Returns landmarks of the first found face or None.

        Args:
            frame: BGR image, shape (H, W, 3)
            timestamp: UNIX timestamp, defaults to current time if None

        Returns:
            FaceLandmarks if face detected, None otherwise
        """
        if self._closed:
            raise RuntimeError("FaceDetector has been closed")
        
        if timestamp is None:
            timestamp = time.time()

        # Conversion to MediaPipe format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Processing based on the selected mode
        if self.is_video_stream:
            # For VIDEO mode, MediaPipe expects the time in milliseconds (integer)
            timestamp_ms = int(timestamp * 1000)
            results = self._detector.detect_for_video(mp_image, timestamp_ms)
        else:
            results = self._detector.detect(mp_image)

        if not results.face_landmarks:
            return None
        
        face_landmarks = results.face_landmarks[0]

        # Conversion to numpy. We crop to 468 points, removing 10 points of the pupils.
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32
        )[:468]

        return FaceLandmarks(
            points=points,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            timestamp=timestamp,
        )
    
    def close(self) -> None:
        """Release MediaPipe resources. Call at completion."""
        if not self._closed:
            self._detector.close()
            self._closed = True

    def __enter__(self) -> "FaceDetector":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()