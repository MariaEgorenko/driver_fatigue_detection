import asyncio
import os
import tempfile
import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager

from backend.app.main import app

@pytest.fixture
def minimal_video_path():
    """Create a minimal test video file (3 frames, 160x120, 1fps)."""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmpfile.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmpfile.name, fourcc, 1.0, (160, 120))

    if not out.isOpened():
        # Fallback if mp4v is not available in CI
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(tmpfile.name, fourcc, 1.0, (160, 120))

    # Write 3 black frames
    for _ in range(3):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        out.write(frame)

    out.release()
    yield tmpfile.name
    
    if os.path.exists(tmpfile.name):
        os.unlink(tmpfile.name)


@pytest_asyncio.fixture
async def e2e_client():
    """Client with real ML pipeline initialized via LifespanManager."""
    async with LifespanManager(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), 
            base_url="http://testserver"
        ) as ac:
            yield ac


@pytest.mark.asyncio
async def test_upload_and_poll(e2e_client: AsyncClient, minimal_video_path: str):
    """Test full video upload and polling until completion."""
    
    # 1. Upload video
    with open(minimal_video_path, "rb") as f:
        response = await e2e_client.post(
            "/api/v1/analysis/upload", 
            files={"file": ("test.mp4", f, "video/mp4")}
        )

    assert response.status_code == 202, f"Upload failed: {response.text}"
    data = response.json()
    session_id = data["session_id"]
    assert data["status"] in ["pending", "processing"]

    # 2. Poll until completed
    timeout = 30
    start_time = asyncio.get_event_loop().time()

    completed_data = None

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            pytest.fail(f"Timeout waiting for session {session_id} to complete")

        status_response = await e2e_client.get(f"/api/v1/analysis/{session_id}/status")
        assert status_response.status_code == 200
        
        status_data = status_response.json()

        if status_data["status"] == "completed":
            completed_data = status_data
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Analysis failed with error: {status_data.get('error')}")

        await asyncio.sleep(1)

    # 3. Assertions on completed data
    assert completed_data is not None
    assert completed_data["progress"] == 1.0
    assert completed_data["overall_fatigue_level"] in [
        "alert",
        "mild_fatigue",
        "severe_fatigue",
    ]
    assert completed_data["frames_analyzed"] == 3