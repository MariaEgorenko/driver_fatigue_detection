import pytest
from fastapi.testclient import TestClient
import cv2
import numpy as np

from backend.app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_analyze_frame_black_image(client: TestClient):
    """Test analyzing a black image (no face detected) returns alert level."""
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    success, buffer = cv2.imencode(".jpg", black_frame)
    assert success is True
    jpg_data = buffer.tobytes()

    response = client.post(
        "/api/v1/analysis/frame",
        files={"frame": ("test.jpg", jpg_data, "image/jpeg")},
        data={"session_id": "test-session-123", "timestamp": 1234567890.0},
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Detail: {response.text}"
    
    data = response.json()
    assert "fatigue_score" in data
    assert data["fatigue_score"]["level"] == "alert"
    
    assert "session_id" in data

def test_analyze_frame_invalid_format(client: TestClient):
    """Test that uploading a non-image file returns 400."""
    response = client.post(
        "/api/v1/analysis/frame",
        files={"frame": ("test.txt", b"not an image", "text/plain")},
        data={"session_id": "test-session-123"},
    )

    assert response.status_code == 400
    assert "Failed to decode image" in response.json()["detail"]

def test_analyze_frame_missing_session_id(client: TestClient):
    """Test that missing session_id returns 422."""
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", black_frame)
    jpg_data = buffer.tobytes()

    response = client.post(
        "/api/v1/analysis/frame", 
        files={"frame": ("test.jpg", jpg_data, "image/jpeg")}
    )

    assert response.status_code == 422
    response_data = response.json()
    
    detail = response_data["detail"]
    assert any(error["loc"] == ["body", "session_id"] for error in detail)
