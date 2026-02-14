"""
Drawing Animation API Routes

Endpoints for animating static drawings/images using:
- CogVideoX-5b-I2V (text-guided animation)
- Stable Video Diffusion XT 1.1 (motion animation)
"""

import os
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services.models.drawing_animation_service import (
    get_drawing_animation_service,
    I2VModel
)
from app.services.minio_client import get_minio_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/drawing-animation", tags=["Drawing Animation"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DrawingAnimationRequest(BaseModel):
    """Request model for drawing animation"""
    drawing_url: str = Field(..., description="URL or path to the drawing/image")
    prompt: Optional[str] = Field(None, description="Text description for animation (optional)")
    model: I2VModel = Field(I2VModel.AUTO, description="Which model to use")
    num_frames: int = Field(25, ge=16, le=49, description="Number of frames")
    fps: int = Field(6, ge=4, le=30, description="Frames per second")
    guidance_scale: float = Field(6.0, ge=1.0, le=20.0, description="CFG scale (CogVideoX only)")
    motion_bucket_id: int = Field(127, ge=0, le=255, description="Motion amount (SVD only)")


class DrawingAnimationResponse(BaseModel):
    """Response model for drawing animation"""
    job_id: str
    status: str
    message: str
    video_url: Optional[str] = None
    model_used: Optional[str] = None
    metadata: Optional[dict] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/upload", response_model=DrawingAnimationResponse)
async def upload_and_animate_drawing(
    file: UploadFile = File(..., description="Drawing/image file"),
    prompt: Optional[str] = Form(None, description="Animation prompt"),
    model: str = Form("auto", description="Model to use: auto/cogvideox/svd"),
    num_frames: int = Form(25, description="Number of frames"),
    fps: int = Form(6, description="Frames per second"),
    guidance_scale: float = Form(6.0, description="CFG scale"),
    motion_bucket_id: int = Form(127, description="Motion amount"),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a drawing/image and generate an animated video

    Workflow:
    1. Upload drawing to MinIO
    2. Select appropriate model (CogVideoX or SVD)
    3. Generate animated video
    4. Return video URL

    Args:
        file: Image file (PNG, JPG, etc.)
        prompt: Optional text description for animation
        model: Which model to use (auto/cogvideox/svd)
        num_frames: Number of frames to generate
        fps: Video framerate
        guidance_scale: Classifier-free guidance (CogVideoX)
        motion_bucket_id: Motion intensity (SVD)

    Returns:
        Job information with video URL when complete
    """
    logger.info(f"📤 Drawing animation request received: {file.filename}")

    try:
        # Generate unique job ID
        job_id = str(uuid4())

        # Upload file to MinIO
        minio_client = get_minio_client()
        bucket_name = "dance-drawings"

        # Create bucket if not exists
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Upload file
        file_extension = os.path.splitext(file.filename)[1]
        minio_path = f"{job_id}/{file.filename}"

        file_content = await file.read()
        from io import BytesIO
        minio_client.put_object(
            bucket_name,
            minio_path,
            BytesIO(file_content),
            length=len(file_content),
            content_type=file.content_type,
        )

        logger.info(f"✅ Uploaded to MinIO: {bucket_name}/{minio_path}")

        # Download to temp file for processing
        temp_dir = "/tmp/dance_drawings"
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, f"{job_id}{file_extension}")

        with open(local_path, "wb") as f:
            f.write(file_content)

        # Generate video
        service = get_drawing_animation_service()

        output_path = os.path.join(temp_dir, f"{job_id}_animated.mp4")

        result = await service.generate_video(
            image_path=local_path,
            prompt=prompt,
            output_path=output_path,
            model=I2VModel(model),
            num_frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            motion_bucket_id=motion_bucket_id,
        )

        if result.get("success"):
            # Upload result to MinIO
            output_minio_path = f"{job_id}/output.mp4"
            with open(result["output_path"], "rb") as f:
                video_content = f.read()
                minio_client.put_object(
                    "dance-videos",
                    output_minio_path,
                    BytesIO(video_content),
                    length=len(video_content),
                    content_type="video/mp4",
                )

            video_url = f"/api/drawing-animation/video/{job_id}"

            # Cleanup temp files
            os.remove(local_path)
            os.remove(result["output_path"])

            return DrawingAnimationResponse(
                job_id=job_id,
                status="completed",
                message="Animation generated successfully!",
                video_url=video_url,
                model_used=result.get("model_used"),
                metadata=result.get("metadata"),
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Animation failed: {result.get('error')}",
            )

    except Exception as e:
        logger.error(f"❌ Drawing animation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/{job_id}")
async def get_animated_video(job_id: str):
    """
    Download generated animated video

    Args:
        job_id: Job ID from upload request

    Returns:
        Video file
    """
    try:
        minio_client = get_minio_client()
        minio_path = f"{job_id}/output.mp4"

        # Download from MinIO
        temp_path = f"/tmp/{job_id}.mp4"
        minio_client.fget_object("dance-videos", minio_path, temp_path)

        return FileResponse(
            temp_path,
            media_type="video/mp4",
            filename=f"animated_{job_id}.mp4",
        )

    except Exception as e:
        logger.error(f"❌ Failed to retrieve video: {e}")
        raise HTTPException(status_code=404, detail="Video not found")


@router.get("/models")
async def list_available_models():
    """
    List available image-to-video models

    Returns:
        List of models with descriptions and requirements
    """
    return {
        "models": [
            {
                "id": "cogvideox",
                "name": "CogVideoX-5b-I2V",
                "description": "Text-guided animation with LangGraph agents",
                "features": ["Text prompts", "High quality", "LangGraph orchestration"],
                "vram_required": "12GB+",
                "speed": "~5-8 min on T4",
            },
            {
                "id": "svd",
                "name": "Stable Video Diffusion XT 1.1",
                "description": "Production-ready motion animation",
                "features": ["Motion-based", "Reliable", "Memory efficient"],
                "vram_required": "8GB+",
                "speed": "~2-3 min on T4",
            },
            {
                "id": "auto",
                "name": "Automatic Selection",
                "description": "Intelligently chooses best model",
                "features": ["Smart selection", "Optimal performance"],
                "vram_required": "8GB+",
                "speed": "Varies",
            },
        ]
    }


@router.get("/presets")
async def list_animation_presets():
    """
    List preset animation styles/prompts

    Returns:
        List of preset configurations
    """
    return {
        "presets": [
            {
                "name": "Gentle Movement",
                "prompt": "gentle subtle movement, calm atmosphere",
                "motion_bucket_id": 80,
                "description": "Subtle, calm animation",
            },
            {
                "name": "Dynamic Action",
                "prompt": "dynamic energetic movement, vibrant action",
                "motion_bucket_id": 180,
                "description": "Energetic, active animation",
            },
            {
                "name": "Dreamy Float",
                "prompt": "floating dreamy movement, ethereal atmosphere",
                "motion_bucket_id": 100,
                "description": "Soft, floating motion",
            },
            {
                "name": "Natural Breeze",
                "prompt": "natural wind-blown movement, outdoor scene",
                "motion_bucket_id": 127,
                "description": "Natural, organic movement",
            },
        ]
    }
