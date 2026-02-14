"""
Hybrid Drawing Animation Service

Intelligently chooses between:
1. CogVideoX-5b-I2V (with LangGraph agents) - Best for text-guided animation
2. Stable Video Diffusion XT 1.1 - Production-ready, reliable

Auto-selects based on:
- Prompt presence (CogVideoX if prompt provided, SVD otherwise)
- GPU availability
- VRAM constraints
"""

import os
import logging
from typing import Optional, Dict, Any, Literal
from enum import Enum

import torch

from .cogvideox_service import get_cogvideox_service
from .svd_service import get_svd_service

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL SELECTION ENUM
# ============================================================================

class I2VModel(str, Enum):
    """Image-to-Video model options"""
    COGVIDEOX = "cogvideox"
    SVD = "svd"
    AUTO = "auto"


# ============================================================================
# HYBRID DRAWING ANIMATION SERVICE
# ============================================================================

class DrawingAnimationService:
    """
    Hybrid service for animating drawings/images

    Features:
    - Automatic model selection
    - Text-guided animation (CogVideoX)
    - Motion-only animation (SVD)
    - Memory-efficient execution
    """

    def __init__(self):
        """Initialize hybrid service"""
        self.cogvideox_service = None
        self.svd_service = None
        self._gpu_available = torch.cuda.is_available()
        self._vram_gb = self._get_vram_gb() if self._gpu_available else 0

        logger.info(f"🎨 Drawing Animation Service initialized")
        logger.info(f"   GPU Available: {self._gpu_available}")
        logger.info(f"   VRAM: {self._vram_gb}GB")

    def _get_vram_gb(self) -> float:
        """Get available VRAM in GB"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    def _select_model(
        self,
        prompt: Optional[str],
        preferred_model: I2VModel,
    ) -> I2VModel:
        """
        Intelligently select which model to use

        Selection logic:
        - If prompt provided and enough VRAM -> CogVideoX
        - If no prompt or low VRAM -> SVD
        - User preference overrides auto-selection
        """
        if preferred_model != I2VModel.AUTO:
            logger.info(f"✅ User selected model: {preferred_model}")
            return preferred_model

        # Auto-selection logic
        has_prompt = bool(prompt and prompt.strip())

        if has_prompt and self._vram_gb >= 12:
            # CogVideoX is better with text prompts and needs ~12GB+ VRAM
            logger.info("🤖 Auto-selected: CogVideoX (text-guided, 12GB+ VRAM)")
            return I2VModel.COGVIDEOX

        elif self._vram_gb >= 8:
            # SVD works well with 8GB+ VRAM
            logger.info("🎬 Auto-selected: SVD (production-ready, 8GB VRAM)")
            return I2VModel.SVD

        else:
            # Fall back to SVD with CPU offloading
            logger.info("⚠️  Auto-selected: SVD with CPU offloading (low VRAM)")
            return I2VModel.SVD

    async def generate_video(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        model: I2VModel = I2VModel.AUTO,
        num_frames: int = 25,
        fps: int = 6,
        guidance_scale: float = 6.0,
        motion_bucket_id: int = 127,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from drawing/image

        Args:
            image_path: Path to input image/drawing
            prompt: Optional text description for animation (CogVideoX only)
            output_path: Output video path
            model: Which model to use (auto/cogvideox/svd)
            num_frames: Number of frames to generate
            fps: Frames per second
            guidance_scale: CFG scale for CogVideoX
            motion_bucket_id: Motion amount for SVD (0-255)
            **kwargs: Additional model-specific parameters

        Returns:
            Dict with success, output_path, model_used, and metadata
        """
        logger.info("="*60)
        logger.info("🎨 DRAWING ANIMATION REQUEST")
        logger.info(f"   Image: {image_path}")
        logger.info(f"   Prompt: {prompt or 'None (motion-only)'}")
        logger.info(f"   Model Preference: {model}")
        logger.info("="*60)

        # Select model
        selected_model = self._select_model(prompt, model)

        try:
            # Generate video with selected model
            if selected_model == I2VModel.COGVIDEOX:
                result = await self._generate_with_cogvideox(
                    image_path=image_path,
                    prompt=prompt or "natural movement",
                    output_path=output_path,
                    num_frames=num_frames,
                    fps=fps,
                    guidance_scale=guidance_scale,
                    **kwargs
                )
            else:  # SVD
                result = await self._generate_with_svd(
                    image_path=image_path,
                    output_path=output_path,
                    num_frames=num_frames,
                    fps=fps,
                    motion_bucket_id=motion_bucket_id,
                    **kwargs
                )

            # Add model info to result
            if result.get("success"):
                result["model_used"] = selected_model.value

            return result

        except Exception as e:
            logger.error(f"❌ Drawing animation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_attempted": selected_model.value,
            }

    async def _generate_with_cogvideox(
        self,
        image_path: str,
        prompt: str,
        output_path: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video using CogVideoX with LangGraph agents"""
        if self.cogvideox_service is None:
            self.cogvideox_service = get_cogvideox_service()

        logger.info("🤖 Using CogVideoX-5b-I2V with LangGraph agents...")
        return await self.cogvideox_service.generate_video(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            **kwargs
        )

    async def _generate_with_svd(
        self,
        image_path: str,
        output_path: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video using Stable Video Diffusion XT 1.1"""
        if self.svd_service is None:
            self.svd_service = get_svd_service()

        logger.info("🎬 Using Stable Video Diffusion XT 1.1...")
        return await self.svd_service.generate_video(
            image_path=image_path,
            output_path=output_path,
            **kwargs
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_drawing_animation_service: Optional[DrawingAnimationService] = None


def get_drawing_animation_service() -> DrawingAnimationService:
    """Get singleton drawing animation service"""
    global _drawing_animation_service
    if _drawing_animation_service is None:
        _drawing_animation_service = DrawingAnimationService()
    return _drawing_animation_service
