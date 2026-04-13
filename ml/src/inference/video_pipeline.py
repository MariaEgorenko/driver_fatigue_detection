import cv2
import pickle
import time
import logging
from typing import Union, Iterator, Optional

from ml.src.types import FrameResult, FeatureWindow
from ml.src.features.face_detector import FaceDetector
from ml.src.features.eye_features import EyeAnalyzer
from ml.src.features.mouth_features import MouthAnalyzer
from ml.src.features.head_pose import HeadPoseEstimator
from ml.src.features.presence_tracker import FacePresenceTracker
from ml.src.inference.fatigue_scorer import FatigueScorer

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Frame-by-frame analysis of the video stream."""

    def __init__(self, config) -> None:
        self._config = config
        self._detector = FaceDetector(is_video_stream=True)
        self._eye = EyeAnalyzer(config)
        self._mouth = MouthAnalyzer(config)
        self._head = HeadPoseEstimator()
        self._presence = FacePresenceTracker(config.FACE_ABSENT_THRESHOLD_SEC)
        self._scorer = FatigueScorer(config)
        
        self._window = FeatureWindow()
        self._frame_counter = 0
        self._last_result: Optional[FrameResult] = None

    def process_frame(
        self,
        frame,
        timestamp: Optional[float] = None,
    ) -> FrameResult:
        """Process one frame."""
        if timestamp is None:
            timestamp = time.time()

        self._frame_counter += 1
        
        # Skipping frames for optimization
        process_this_frame = (
            self._frame_counter == 1
            or self._frame_counter % self._config.FRAME_PROCESSING_INTERVAL == 0
        )

        if not process_this_frame:
            if self._last_result is None:
                self._last_result = FrameResult(
                    timestamp=timestamp,
                    frame_index=self._frame_counter,
                    face_detected=False,
                    fatigue_score=None,
                )
            return self._last_result

        landmarks = self._detector.detect(frame, timestamp=timestamp)

        if landmarks is not None:
            face_detected = True

            ear = self._eye.compute_ear(landmarks)
            mar = self._mouth.compute_mar(landmarks)
            head_pose = self._head.estimate(landmarks)
            
            logger.debug(f"[FRAME] mar={mar:.3f}, ear={ear:.3f}")

            self._window.ear_history.append(ear)
            self._window.mar_history.append(mar)
            self._window.timestamps.append(timestamp)
            self._window.face_detected_flags.append(True)

            if head_pose is not None:
                self._window.head_poses.append(head_pose)

            # Event detection
            blinks = self._eye.detect_blinks(self._window)
            if blinks:
                self._window.blink_events.extend(blinks)

            yawns = self._mouth.detect_yawn(self._window)
            if yawns:
                self._window.yawn_events.extend(yawns)

            absence_event = self._presence.update(face_detected=True, timestamp=timestamp)
            if absence_event is not None:
                self._window.absence_events.append(absence_event)
                
        else:
            # Face was not found
            face_detected = False
            self._window.face_detected_flags.append(False)
            self._window.timestamps.append(timestamp)
            self._window.ear_history.append(0.0)
            self._window.mar_history.append(0.0)

            absence_event = self._presence.update(face_detected=False, timestamp=timestamp)
            if absence_event is not None:
                self._window.absence_events.append(absence_event)

        self._trim_window()

        fatigue_score = self._scorer.score(self._window)

        result = FrameResult(
            timestamp=timestamp,
            frame_index=self._frame_counter,
            face_detected=face_detected,
            fatigue_score=fatigue_score,
            landmarks=landmarks,
        )

        self._last_result = result
        return result
    
    def _trim_window(self):
        """Slice the window to the maximum length MAX_WINDOW_SEC."""
        # Storing data with a margin for PERCLOS (which looks at 1 minute ago)
        MAX_WINDOW_SEC = max(60, getattr(self._config, 'PERCLOS_WINDOW_SEC', 60))
        
        if not self._window.timestamps:
            return

        cutoff_time = self._window.timestamps[-1] - MAX_WINDOW_SEC

        keep_from = len(self._window.timestamps)
        
        for i, ts in enumerate(self._window.timestamps):
            if ts >= cutoff_time:
                keep_from = i
                break
                
        if keep_from == 0:
            return
            
        if keep_from == len(self._window.timestamps):
            self.reset_session()
            return

        self._window.ear_history = self._window.ear_history[keep_from:]
        self._window.mar_history = self._window.mar_history[keep_from:]
        self._window.timestamps = self._window.timestamps[keep_from:]
        self._window.face_detected_flags = self._window.face_detected_flags[keep_from:]
        
        # head_poses can have a different length (added only if face_detected)
        # so we filter it independently
        # Ideally, the HeadPose should also have a timestamp inside the object.
        # But for now, we just leave the last N elements proportionally
        frames_kept = len(self._window.timestamps)
        if len(self._window.head_poses) > frames_kept:
            self._window.head_poses = self._window.head_poses[-frames_kept:]

        self._window.blink_events = [e for e in self._window.blink_events if e.timestamp >= cutoff_time]
        self._window.yawn_events = [e for e in self._window.yawn_events if e.timestamp >= cutoff_time]
        self._window.absence_events = [e for e in self._window.absence_events if e.start_time >= cutoff_time]

    def process_video(
        self,
        source: Union[str, int],
    ) -> Iterator[FrameResult]:
        """FrameResult generator for a video file or camera."""
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        last_timestamp_sec = -1.0

        if not cap.isOpened():
            raise ValueError(f"Cannot open: {source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                if timestamp_sec <= last_timestamp_sec:
                    timestamp_sec = last_timestamp_sec + (1.0 / fps)

                last_timestamp_sec = timestamp_sec

                result = self.process_frame(frame, timestamp=timestamp_sec)
                yield result
                frame_idx += 1

        finally:
            cap.release()

    def reset_session(self) -> None:
        """Reset the status for a new session."""
        self._window = FeatureWindow()
        self._frame_counter = 0
        self._presence.reset()
        self._last_result = None

    def close(self) -> None:
        """Forcibly release detector resources."""
        if hasattr(self, '_detector') and self._detector is not None:
            self._detector.close()

    def save(self, path: str) -> None:
        """Save the pipeline configuration."""
        with open(path, "wb") as f:
            pickle.dump({"config": self._config}, f)

    @classmethod
    def load(cls, path: str) -> "VideoPipeline":
        """Download the config and initialize the pipeline."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["config"])