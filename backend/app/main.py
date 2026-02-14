"""
Main FastAPI application for Dance Video Generator
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from datetime import datetime
import uuid
import asyncio
import logging

from app.core.config import settings
from app.api.routes import drawing_animation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered dance video generator with facial expressions and text prompts"
)

# Include routers
app.include_router(drawing_animation.router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AnalyzeReferenceRequest(BaseModel):
    """Request to analyze a reference video"""
    video_url: str  # YouTube URL or uploaded file path
    source: str = "youtube"  # "youtube", "upload", "library"


class AnalyzeReferenceResponse(BaseModel):
    """Response from reference video analysis"""
    reference_id: str
    num_dancers: int
    duration_seconds: float
    fps: int
    has_audio: bool
    has_vocals: bool
    has_face_data: bool
    preview_url: Optional[str] = None


class GenerateDanceRequest(BaseModel):
    """Request to generate dance video"""
    # Input photo
    photo_url: str
    photo_type: str = "group"  # "solo" or "group"

    # Reference video
    reference_video_url: str
    reference_source: str = "youtube"

    # Audio
    audio_url: Optional[str] = None
    audio_source: str = "reference_video"  # "reference_video", "youtube", "upload"
    song_title: Optional[str] = None

    # TEXT PROMPTS (Sora/Kling-style) - NEW!
    scene_prompt: Optional[str] = None  # "dancing in a neon-lit nightclub"
    style_prompt: Optional[str] = "cinematic, 4k, dramatic lighting"
    negative_prompt: Optional[str] = "low quality, blurry, distorted"
    background_mode: str = "original"  # "original", "generated", "custom"
    background_url: Optional[str] = None

    # Choreography
    choreography_strategy: str = "all_sync"
    choreography_mapping: Optional[dict] = None

    # Expression settings
    enable_facial_expressions: bool = True
    enable_lip_sync: bool = True
    expression_intensity: float = 1.0

    # Model selection
    body_motion_model: str = "animatediff"  # "magicanimate", "magicdance", "animatediff"
    face_expression_model: str = "liveportrait"  # "fomm", "liveportrait"


class GenerateDanceResponse(BaseModel):
    """Response from generation request"""
    project_id: str
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    project_id: str
    status: str
    progress: int  # 0-100
    current_step: str
    logs: List[str]
    final_video_url: Optional[str] = None
    preview_url: Optional[str] = None
    estimated_time_remaining: Optional[int] = None  # seconds


class StylePresetResponse(BaseModel):
    """Style preset for quick selection"""
    id: str
    name: str
    description: str
    scene_prompt_template: str
    style_prompt: str
    category: str
    preview_url: Optional[str] = None


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    # TODO: Check database, Redis, MinIO connectivity
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "redis": "connected",
            "minio": "connected"
        }
    }


@app.post("/api/v1/dance/reference/analyze", response_model=AnalyzeReferenceResponse)
async def analyze_reference_video(request: AnalyzeReferenceRequest):
    """
    Analyze a reference dance video

    - Downloads video if YouTube URL
    - Detects number of dancers
    - Extracts poses and facial expressions
    - Returns metadata

    This is a background task that can take 30-60 seconds
    """
    try:
        from app.services.youtube_service import get_youtube_service

        reference_id = str(uuid.uuid4())
        youtube_service = get_youtube_service()

        # Get video info without downloading
        video_info = youtube_service.get_video_info(request.video_url)

        logger.info(f"✅ Analyzed reference: {video_info['title']}")

        return AnalyzeReferenceResponse(
            reference_id=reference_id,
            num_dancers=1,  # TODO: Implement multi-person detection
            duration_seconds=video_info['duration'],
            fps=video_info['fps'],
            has_audio=True,
            has_vocals=True,  # Assume yes for YouTube videos
            has_face_data=True,  # Will be extracted during generation
            preview_url=video_info.get('thumbnail')
        )

    except Exception as e:
        logger.error(f"Error analyzing reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/dance/upload/photo")
async def upload_photo(file: UploadFile = File(...)):
    """
    Upload a photo (solo or group)

    Returns:
        - Photo URL (MinIO path)
        - Detected persons count
        - Preview with bounding boxes
    """
    try:
        from app.services.minio_client import get_minio_client
        from io import BytesIO

        # Generate unique ID
        photo_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        object_name = f"{photo_id}.{file_extension}"

        # Read file content
        file_content = await file.read()

        # Save to MinIO
        minio_client = get_minio_client()
        bucket_name = "dance-photos"

        # Ensure bucket exists
        if not minio_client.client.bucket_exists(bucket_name):
            minio_client.client.make_bucket(bucket_name)

        # Upload to MinIO
        minio_client.client.put_object(
            bucket_name,
            object_name,
            BytesIO(file_content),
            length=len(file_content),
            content_type=file.content_type or "image/jpeg"
        )

        photo_url = f"minio://{bucket_name}/{object_name}"

        logger.info(f"✅ Photo uploaded: {photo_url}")

        return {
            "photo_url": photo_url,
            "detected_persons": 1,  # TODO: Implement person detection
            "preview_url": f"/api/v1/photos/{photo_id}/preview"
        }

    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/dance/generate", response_model=GenerateDanceResponse)
async def generate_dance_video(
    request: GenerateDanceRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate dance video with facial expressions and text prompts

    Pipeline:
    1. Process reference video (poses + expressions)
    2. Process input photo (detect persons + faces)
    3. Map choreography
    4. Generate body motion (AnimateDiff with pose ControlNet + scene prompt)
    5. Generate facial expressions (FOMM/LivePortrait)
    6. Composite face onto body
    7. Generate/replace background (if requested)
    8. Final composition with audio

    Returns immediately with job_id for progress tracking
    """
    try:
        from app.tasks.dance_generation_task import generate_dance_video_task

        project_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        logger.info(f"🎬 Starting generation for project {project_id}")
        logger.info(f"   Photo: {request.photo_url}")
        logger.info(f"   Reference: {request.reference_video_url}")
        logger.info(f"   Scene prompt: {request.scene_prompt}")
        logger.info(f"   Style prompt: {request.style_prompt}")
        logger.info(f"   Background mode: {request.background_mode}")

        # Queue Celery task
        task = generate_dance_video_task.apply_async(
            args=[project_id, request.dict()],
            task_id=job_id
        )

        logger.info(f"✅ Queued task {job_id}")

        return GenerateDanceResponse(
            project_id=project_id,
            job_id=job_id,
            status="queued"
        )

    except Exception as e:
        logger.error(f"Error starting generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dance/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get job status (Server-Sent Events for real-time updates)

    Client can listen for updates:
    ```javascript
    const eventSource = new EventSource('/api/v1/dance/status/' + jobId);
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Progress:', data.progress);
    };
    ```
    """
    from app.celery_app import celery_app
    from celery.result import AsyncResult

    async def event_generator():
        """Generate SSE events for job progress"""
        task = AsyncResult(job_id, app=celery_app)

        last_progress = -1

        while True:
            # Get task state
            state = task.state
            info = task.info if task.info else {}

            # Extract progress info
            if state == 'PENDING':
                progress = 0
                current_step = "Queued"
                logs = ["Task is queued..."]
            elif state == 'PROGRESS':
                progress = info.get('progress', 0)
                current_step = info.get('step', 'Processing')
                logs = info.get('logs', [])
            elif state == 'SUCCESS':
                progress = 100
                current_step = "Complete"
                logs = ["Video generation complete!"]
            elif state == 'FAILURE':
                progress = 0
                current_step = "Failed"
                logs = [f"Error: {str(task.info)}"]
            else:
                progress = 0
                current_step = state
                logs = []

            # Only send update if progress changed
            if progress != last_progress or state in ['SUCCESS', 'FAILURE']:
                status_data = {
                    "job_id": job_id,
                    "project_id": job_id,  # Using job_id as project_id for simplicity
                    "status": state.lower(),
                    "progress": progress,
                    "current_step": current_step,
                    "logs": logs,
                }

                # Add final video URL if complete
                if state == 'SUCCESS' and task.result:
                    final_video_url = task.result.get('final_video_url')
                    if final_video_url:
                        # Convert MinIO URL to download endpoint
                        video_filename = final_video_url.split('/')[-1]
                        status_data["final_video_url"] = f"/api/v1/dance/video/{job_id}"

                yield {
                    "event": "progress" if state == 'PROGRESS' else state.lower(),
                    "data": status_data
                }

                last_progress = progress

                # Exit if done
                if state in ['SUCCESS', 'FAILURE']:
                    break

            await asyncio.sleep(1)  # Check every second

    return EventSourceResponse(event_generator())


@app.get("/api/v1/dance/video/{project_id}")
async def download_video(project_id: str):
    """
    Get final video for download
    """
    try:
        from fastapi.responses import FileResponse
        from app.services.minio_client import get_minio_client
        import os

        minio_client = get_minio_client()

        # Try to find the video in MinIO
        object_name = f"{project_id}_final.mp4"

        # Download from MinIO to temp file
        temp_path = f"/tmp/{project_id}.mp4"
        try:
            minio_client.client.fget_object(
                "dance-videos",
                object_name,
                temp_path
            )

            return FileResponse(
                temp_path,
                media_type="video/mp4",
                filename=f"dance_{project_id}.mp4"
            )

        except Exception as e:
            logger.error(f"Video not found in MinIO: {e}")
            raise HTTPException(status_code=404, detail="Video not found")

    except Exception as e:
        logger.error(f"Error getting video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/style-presets", response_model=List[StylePresetResponse])
async def get_style_presets():
    """
    Get pre-defined style presets (Sora/Kling-style)

    Examples:
    - Cyberpunk Nightclub
    - Beach Sunset
    - Urban Rooftop
    - Fantasy Forest
    - Retro 80s Disco
    """
    presets = [
        StylePresetResponse(
            id="1",
            name="Cyberpunk Nightclub",
            description="Neon-lit futuristic nightclub with dynamic lighting",
            scene_prompt_template="dancing in a cyberpunk nightclub with neon lights and holographic displays",
            style_prompt="cinematic, 4k, dramatic neon lighting, cyberpunk aesthetic, high contrast",
            category="nightclub",
            preview_url="/static/presets/cyberpunk.jpg"
        ),
        StylePresetResponse(
            id="2",
            name="Beach Sunset",
            description="Golden hour beach with waves and warm lighting",
            scene_prompt_template="dancing on a beach at sunset with golden light and ocean waves",
            style_prompt="cinematic, 4k, warm golden hour lighting, beautiful scenery, professional photography",
            category="nature",
            preview_url="/static/presets/beach.jpg"
        ),
        StylePresetResponse(
            id="3",
            name="Urban Rooftop",
            description="City skyline at night with lights",
            scene_prompt_template="dancing on an urban rooftop with city skyline at night",
            style_prompt="cinematic, 4k, city lights, dramatic urban photography, bokeh background",
            category="urban",
            preview_url="/static/presets/rooftop.jpg"
        ),
        StylePresetResponse(
            id="4",
            name="Retro 80s Disco",
            description="Colorful 80s disco with disco ball and vibrant colors",
            scene_prompt_template="dancing in a retro 80s disco with disco ball and colorful lights",
            style_prompt="vibrant colors, 80s aesthetic, disco lighting, retro style, high energy",
            category="nightclub",
            preview_url="/static/presets/disco.jpg"
        ),
        StylePresetResponse(
            id="5",
            name="Fantasy Forest",
            description="Magical forest with glowing particles and ethereal atmosphere",
            scene_prompt_template="dancing in a magical forest with glowing fireflies and ethereal atmosphere",
            style_prompt="cinematic, 4k, fantasy lighting, magical atmosphere, dreamy, enchanted",
            category="fantasy",
            preview_url="/static/presets/forest.jpg"
        )
    ]

    return presets


@app.get("/api/v1/templates", response_model=List[dict])
async def get_choreography_templates():
    """
    Get pre-analyzed reference videos (template library)

    Popular dances ready to use:
    - BTS Dynamite
    - Blackpink DDU-DU DDU-DU
    - Michael Jackson Thriller
    - etc.
    """
    # TODO: Query database for public templates
    return [
        {
            "id": "1",
            "name": "BTS - Dynamite (Chorus)",
            "num_dancers": 7,
            "duration": 30,
            "difficulty": "medium",
            "tags": ["kpop", "energetic", "synchronized"],
            "preview_url": "/static/templates/bts-dynamite.mp4"
        },
        {
            "id": "2",
            "name": "Michael Jackson - Thriller",
            "num_dancers": 1,
            "duration": 45,
            "difficulty": "hard",
            "tags": ["classic", "iconic", "solo"],
            "preview_url": "/static/templates/mj-thriller.mp4"
        }
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
