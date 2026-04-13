import time
import os
import shutil
import tempfile
import logging
import cv2
import numpy as np
import asyncio
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.dependencies import get_ml_pipeline
from backend.app.core.database import get_db, AsyncSessionLocal
from backend.app.core.config import get_settings
from backend.app.core.metrics import (
    frame_processing_duration,
    active_sessions,
    fatigue_events_total,
)
from backend.app.schemas.analysis import (
    FrameAnalysisResponse,
    FatigueScoreSchema,
    FatigueEventSchema,
    FaceLandmarksSchema,
    AnalysisSessionStatus,
)
from backend.app.services.analysis_service import AnalysisService

settings = get_settings()

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_analysis_service(db: AsyncSession = Depends(get_db)) -> AnalysisService:
    """Dependency to get AnalysisService instance."""
    return AnalysisService(db)

def convert_fatigue_score_to_schema(score) -> FatigueScoreSchema:
    events = [
        FatigueEventSchema(
            event_type=event.event_type,
            timestamp_sec=event.timestamp,
            duration_sec=event.duration_sec,
            severity=event.severity,
            metadata=event.metadata,
        )
        for event in score.events
    ]
    return FatigueScoreSchema(
        level=score.level,
        confidence=score.confidence,
        perclos=score.perclos,
        blink_rate=score.blink_rate,
        yawn_count=score.yawn_count,
        head_down=score.head_down,
        face_absent=score.face_absent,
        events=events,
    )

def convert_landmarks_to_schema(landmarks) -> FaceLandmarksSchema | None:
    if landmarks is None:
        return None
    return FaceLandmarksSchema(
        points=landmarks.points.tolist(),
        frame_width=landmarks.frame_width,
        frame_height=landmarks.frame_height,
    )


@router.post("/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(
    frame: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    session_id: str = Form(..., description="Session ID for tracking"),
    timestamp: float = Form(default=None, description="Unix timestamp of the frame"),
    pipeline = Depends(get_ml_pipeline),
):
    """Real-time single frame analysis."""
    start_time = time.time()

    try:
        contents = await frame.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File size exceeds 10MB")

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        if timestamp is None:
            timestamp = time.time()

        result = await asyncio.to_thread(pipeline.process_frame, img, timestamp)

        if result.fatigue_score and result.fatigue_score.events:
            for event in result.fatigue_score.events:
                fatigue_events_total.labels(
                    event_type=event.event_type, severity=event.severity
                ).inc()

        processing_time = time.time() - start_time
        frame_processing_duration.observe(processing_time)

        if result.fatigue_score is None:
            raise HTTPException(status_code=500, detail="Pipeline returned no fatigue score")

        score_schema = convert_fatigue_score_to_schema(result.fatigue_score)
        landmarks_schema = convert_landmarks_to_schema(result.landmarks)

        try:
            session_uuid = UUID(session_id)
        except ValueError:
            session_uuid = uuid4()

        return FrameAnalysisResponse(
            session_id=session_uuid,
            timestamp=result.timestamp,
            frame_index=result.frame_index,
            face_detected=result.face_detected,
            fatigue_score=score_schema,
            landmarks=landmarks_schema,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
def _run_ml_pipeline_sync(pipeline, tmp_path):
    """A synchronous function that will perform heavy ML in a separate thread"""
    return list(pipeline.process_video(tmp_path))

async def _process_video_background(
    session_id: UUID,
    tmp_path: str,
    tmp_dir: str,
    pipeline,
):
    """Background task. Uses ITS OWN DATABASE session."""
    async with AsyncSessionLocal() as db:
        service = AnalysisService(db)
        
        try:
            await service.update_session(session_id, status="processing", progress=0.0)
            await db.commit()
            active_sessions.inc()

            start_time = time.time()
            
            results = await asyncio.to_thread(_run_ml_pipeline_sync, pipeline, tmp_path)
            
            total_frames = len(results)
            events_collected = []
            worst_level = "alert"

            for result in results:
                if result.fatigue_score:
                    score = result.fatigue_score
                    if score.level == "severe_fatigue":
                        worst_level = "severe_fatigue"
                    elif score.level == "mild_fatigue" and worst_level == "alert":
                        worst_level = "mild_fatigue"

                    if score.events:
                        events_collected.extend([
                            {
                                "event_type": e.event_type,
                                "timestamp_sec": e.timestamp,
                                "duration_sec": e.duration_sec,
                                "severity": e.severity,
                                "metadata": e.metadata,
                            }
                            for e in score.events
                        ])

            duration = time.time() - start_time

            await service.save_events(session_id, events_collected)
            await service.update_session(
                session_id,
                status="completed",
                progress=1.0,
                duration_sec=duration,
                overall_fatigue_level=worst_level,
                frames_analyzed=total_frames,
            )
            await db.commit()

            logger.info(f"Completed session {session_id}: {total_frames} frames in {duration:.2f}s")

        except Exception as e:
            logger.error(f"Error processing video for session {session_id}: {e}", exc_info=True)
            await service.update_session(session_id, status="failed", error=str(e))
            await db.commit()
        finally:
            active_sessions.dec()
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/upload", response_model=AnalysisSessionStatus, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, etc.)"),
    pipeline = Depends(get_ml_pipeline),
    service: AnalysisService = Depends(get_analysis_service),
    db: AsyncSession = Depends(get_db),
):
    allowed_types = ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", "video/x-matroska"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported type. Allowed: {allowed_types}")

    session = await service.create_session(source_type="video", source_name=file.filename)
    await db.commit() 

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        file_size = os.path.getsize(tmp_path)
        if file_size > settings.VIDEO_MAX_SIZE_MB * 1024 * 1024:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(status_code=413, detail=f"File exceeds {settings.VIDEO_MAX_SIZE_MB}MB limit")

    finally:
        await file.close()

    background_tasks.add_task(
        _process_video_background,
        session_id=session.id,
        tmp_path=tmp_path,
        tmp_dir=tmp_dir,
        pipeline=pipeline,
    )

    return session


@router.get("/{session_id}/status", response_model=AnalysisSessionStatus)
async def get_session_status(
    session_id: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session