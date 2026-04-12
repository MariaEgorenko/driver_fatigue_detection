import cv2
import numpy as np
import pytest
import time
from pathlib import Path

from ml.src.features.face_detector import FaceDetector
from ml.src.types import FaceLandmarks

# Paths to test data
FIXTURES_DIR = Path("ml/tests/fixtures")
FACES_DIR = FIXTURES_DIR / "faces"
NO_FACES_DIR = FIXTURES_DIR / "no_faces"


class TestFaceDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        """The fixture creates one instance of the detector for the entire class of tests."""
        with FaceDetector(is_video_stream=False) as det:
            yield det

    def test_detect_returns_none_on_blank_frame(self, detector):
        """Test that blank (black) frame returns None."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank_frame)
        assert result is None

    def test_detect_with_faces(self, detector):
        """We check that the photo from the faces folder contains stable faces."""
        if not FACES_DIR.exists():
            pytest.skip(f"Directory {FACES_DIR} not found")
            
        image_files = list(FACES_DIR.glob("*.*"))
        if not image_files:
            pytest.skip("No images in faces fixtures")

        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            assert frame is not None, f"Could not load {img_path}"
            
            result = detector.detect(frame)
            
            # The face must be found
            assert result is not None, f"Face not detected in {img_path.name}"
            assert isinstance(result, FaceLandmarks)
            
            # Checking the exact dimension
            assert result.points.shape == (468, 3)
            
            # Checking the normalization of the X and Y coordinates
            assert np.all((result.points[:, :2] >= 0.0) & (result.points[:, :2] <= 1.0))

    def test_detect_without_faces(self, detector):
        """We check that there are NO faces in the photo from the no_faces/ folder."""
        if not NO_FACES_DIR.exists():
            pytest.skip(f"Directory {NO_FACES_DIR} not found")
            
        image_files = list(NO_FACES_DIR.glob("*.*"))
        if not image_files:
            pytest.skip("No images in no_faces fixtures")

        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            assert frame is not None, f"Could not load {img_path}"
            
            result = detector.detect(frame)
            assert result is None, f"False positive face detected in {img_path.name}"

    def test_detect_timestamp_default(self, detector):
        """We check that if the timestamp is not transmitted, the current time is generated."""
        if not FACES_DIR.exists() or not list(FACES_DIR.glob("*.*")):
            pytest.skip("Need a face image to test timestamp logic")
            
        img_path = list(FACES_DIR.glob("*.*"))[0]
        frame = cv2.imread(str(img_path))
        
        time_before = time.time()
        result = detector.detect(frame)
        time_after = time.time()
        
        assert result is not None
        assert time_before <= result.timestamp <= time_after

    def test_context_manager(self):
        """Test that context manager calls close on exit."""
        detector = FaceDetector(is_video_stream=False)

        with detector as det:
            assert det is detector
            assert not detector._closed

        assert detector._closed