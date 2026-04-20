import time
import os
from pathlib import Path
import shutil
import tempfile
import logging
import cv2
import numpy as np
import asyncio
from uuid import UUID, uuid4

from fastapi import (
    APIRouter, Depends, UploadFile,
    File, Form, HTTPException,
    BackgroundTasks, Request,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.dependencies import get_ml_pipeline, get_streaming_pipeline
from backend.app.core.database import get_db
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
    request: Request,
    frame: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    session_id: str | None = Form(default=None, description="Session ID for tracking"),
    timestamp: float | None = Form(default=None, description="Unix timestamp of the frame")
):
    """Real-time single frame analysis."""
    if not session_id:
        session_id = str(uuid4())

    pipeline = get_streaming_pipeline(session_id, request)

    start_time = time.time()

    try:
        MAX_FRAME_SIZE = settings.MAX_FRAME_SIZE_MB * 1024 * 1024
        chunks = []
        total = 0
        while chunk := await frame.read(65536):
            total += len(chunk)
            if total > MAX_FRAME_SIZE:
                raise HTTPException(status_code=413, detail="File size exceeds 10MB")
            chunks.append(chunk)
        
        contents = b"".join(chunks)

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

        if session_id:
            try:
                session_uuid = UUID(session_id)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Invalid session_id format: '{session_id}'")
        else:
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
        raise HTTPException(status_code=500, detail=f"Internal server error")    

async def _process_video_background(
    session_id: UUID,
    tmp_path: str,
    tmp_dir: str,
    pipeline,
    db_sessionmaker,
):
    """Background task. Uses ITS OWN DATABASE session."""
    if db_sessionmaker is None:
        logger.error("No database sessionmaker provided to background task")
        return
    
    async with db_sessionmaker() as db:
        service = AnalysisService(db)
        
        incremented = False
        try:
            await service.update_session(session_id, status="processing", progress=0.0)
            await db.commit()
            active_sessions.inc()
            incremented = True

            start_time = time.time()
            events_collected = []
            worst_level = "alert"
            total_frames = 0

            sync_gen = pipeline.process_video(tmp_path)

            seen_events = set()

            def get_next_chunk(generator, chunk_size=50):
                chunk = []
                try:
                    for _ in range(chunk_size):
                        chunk.append(next(generator))
                except StopIteration:
                    pass
                return chunk
            
            severe_frames_streak = 0
            mild_frames_streak = 0

            while True:
                chunk = await asyncio.to_thread(get_next_chunk, sync_gen, 50)

                if not chunk:
                    break

                for result in chunk:
                    total_frames += 1
                    if result.fatigue_score:
                        score = result.fatigue_score

                        if score.level == "severe_fatigue":
                            severe_frames_streak += 1
                            mild_frames_streak = 0
                            if severe_frames_streak > 30:
                                worst_level = "severe_fatigue"

                        elif score.level == "mild_fatigue" and worst_level == "alert":
                            mild_frames_streak += 1
                            severe_frames_streak = 0
                            if mild_frames_streak > 30 and worst_level == "alert":
                                worst_level = "mild_fatigue"
                        
                        else:
                            severe_frames_streak = 0
                            mild_frames_streak = 0

                        if score.events:
                            for e in score.events:
                                event_key = (e.event_type, e.timestamp)

                                if event_key not in seen_events:
                                    seen_events.add(event_key)
                                    events_collected.append({
                                        "event_type": e.event_type,
                                        "timestamp_sec": e.timestamp,
                                        "duration_sec": e.duration_sec,
                                        "severity": e.severity,
                                        "metadata": e.metadata,
                                    })


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
            await db.rollback()
            await service.update_session(session_id, status="failed", error=str(e))
            await db.commit()
        finally:
            if incremented:
                active_sessions.dec()
            shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/upload", response_model=AnalysisSessionStatus, status_code=202)
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, etc.)"),
    pipeline = Depends(get_ml_pipeline),
    service: AnalysisService = Depends(get_analysis_service),
    db: AsyncSession = Depends(get_db),
):
    allowed_types = ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", "video/x-matroska"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported type. Allowed: {allowed_types}")

    MAX_SIZE_BYTES = settings.VIDEO_MAX_SIZE_MB * 1024 * 1024
    
    content_length = request.headers.get("content-length")
    if content_length is not None:
        if int(content_length) > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=
                    f"File exceeds {settings.VIDEO_MAX_SIZE_MB}MB limit, declared: {int(content_length)/(1024*1024):.1f}MB"
            )

    tmp_dir = tempfile.mkdtemp()
    suffix = Path(file.filename).suffix.lower()
    safe_filename = f"{uuid4().hex}{suffix}"
    tmp_path = os.path.join(tmp_dir, safe_filename)
    
    try:
        chunks = []
        total = 0
        while chunk := await file.read(65536):
            total += len(chunk)
            if total > MAX_SIZE_BYTES:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f"File exceeds {settings.VIDEO_MAX_SIZE_MB}MB limit")
            chunks.append(chunk)
        
        with open(tmp_path, "wb") as f:
            f.write(b"".join(chunks))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {str(e)}")
    finally:
        await file.close()

    session = await service.create_session(source_type="video", source_name=file.filename)
    await db.commit()

    db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)

    background_tasks.add_task(
        _process_video_background,
        session_id=session.id,
        tmp_path=tmp_path,
        tmp_dir=tmp_dir,
        pipeline=pipeline,
        db_sessionmaker=db_sessionmaker,
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