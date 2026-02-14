"""
Stable Video Diffusion XT 1.1 Image-to-Video Service

Production-ready I2V model from Stability AI
Optimized for Colab T4 (8-12GB VRAM)

Model: stabilityai/stable-video-diffusion-img2vid-xt-1-1
Paper: https://arxiv.org/abs/2311.15127
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

logger = logging.getLogger(__name__)


# ============================================================================
# STABLE VIDEO DIFFUSION SERVICE
# ============================================================================

class SVDService:
    """
    Stable Video Diffusion XT 1.1 Service

    Features:
    - Production-ready quality
    - Optimized for 8-12GB VRAM
    - Fast inference (~2-3 min on T4)
    - Reliable and well-maintained
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_model_cpu_offload: bool = True,
    ):
        """
        Initialize SVD pipeline

        Args:
            model_id: HuggingFace model ID
            device: Device to use (cuda/cpu)
            enable_model_cpu_offload: Enable model CPU offloading for low VRAM
        """
        self.model_id = model_id
        self.device = device
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.pipeline = None

        logger.info(f"Initializing SVD XT 1.1 service on {device}")

    def load_model(self) -> None:
        """Load SVD pipeline with optimizations"""
        if self.pipeline is not None:
            logger.info("Model already loaded")
            return

        try:
            logger.info(f"Loading SVD from {self.model_id}...")

            # Load pipeline with half precision
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            # Memory optimizations
            if self.enable_model_cpu_offload and self.device == "cuda":
                logger.info("Enabling model CPU offloading...")
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to(self.device)

            # Enable memory-efficient attention
            self.pipeline.enable_attention_slicing()

            # Enable VAE tiling for lower memory usage
            self.pipeline.enable_vae_slicing()

            logger.info("✅ SVD XT 1.1 model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load SVD model: {e}")
            raise

    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and memory cleared")

    async def generate_video(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        num_frames: int = 25,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 8,
        num_inference_steps: int = 25,
    ) -> Dict[str, Any]:
        """
        Generate video from image using Stable Video Diffusion

        Args:
            image_path: Path to input image
            output_path: Output video path (optional)
            num_frames: Number of frames (default: 25, max: 49)
            fps: Frames per second (default: 6)
            motion_bucket_id: Motion amount (0-255, default: 127)
            noise_aug_strength: Noise augmentation strength (default: 0.02)
            decode_chunk_size: Decode chunk size for memory (default: 8)
            num_inference_steps: Denoising steps (default: 25)

        Returns:
            Dict with output_path, status, and metadata
        """
        logger.info("🎬 Starting SVD XT 1.1 video generation...")

        try:
            # Load model if not loaded
            self.load_model()

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Resize to optimal dimensions (1024x576)
            target_width = 1024
            target_height = 576
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

            logger.info(f"Generating {num_frames} frames @ {fps}fps...")

            # Generate video
            frames = self.pipeline(
                image=image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).frames[0]

            # Export to video
            output_path = output_path or "/tmp/svd_output.mp4"
            export_to_video(frames, output_path, fps=fps)

            logger.info(f"✅ Video generated: {output_path}")

            return {
                "success": True,
                "output_path": output_path,
                "metadata": {
                    "model": self.model_id,
                    "num_frames": num_frames,
                    "fps": fps,
                    "resolution": f"{target_width}x{target_height}",
                    "motion_bucket_id": motion_bucket_id,
                }
            }

        except Exception as e:
            logger.error(f"❌ SVD generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_svd_service: Optional[SVDService] = None


def get_svd_service() -> SVDService:
    """Get singleton SVD service instance"""
    global _svd_service
    if _svd_service is None:
        _svd_service = SVDService()
    return _svd_service
