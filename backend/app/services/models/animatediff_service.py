"""
AnimateDiff Service for Pose-Guided Video Generation

Uses AnimateDiff with ControlNet for pose-guided animation
"""
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    DDIMScheduler,
    ControlNetModel,
)
from diffusers.utils import export_to_video
import cv2

logger = logging.getLogger(__name__)


class AnimateDiffService:
    """
    Service for generating pose-guided videos using AnimateDiff

    Combines:
    - AnimateDiff for temporal consistency
    - ControlNet for pose guidance
    - Text prompts for scene/style control
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        base_model: str = "runwayml/stable-diffusion-v1-5",
    ):
        """
        Initialize AnimateDiff service

        Args:
            device: Device to use (cuda/cpu)
            model_id: AnimateDiff motion adapter model ID
            base_model: Base Stable Diffusion model
        """
        self.device = device
        self.model_id = model_id
        self.base_model = base_model
        self.pipeline = None

        logger.info(f"🎬 Initializing AnimateDiff on {device}")

    def load_model(self):
        """Load AnimateDiff pipeline with ControlNet"""
        if self.pipeline is not None:
            logger.info("AnimateDiff already loaded")
            return

        try:
            logger.info(f"Loading AnimateDiff from {self.model_id}...")

            # Load motion adapter
            adapter = MotionAdapter.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            # Load ControlNet for pose
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            # Load AnimateDiff pipeline
            self.pipeline = AnimateDiffPipeline.from_pretrained(
                self.base_model,
                motion_adapter=adapter,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            # Optimize for memory
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_vae_slicing()
            else:
                self.pipeline.to(self.device)

            # Use efficient scheduler
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config,
                timestep_spacing="trailing",
                beta_schedule="linear",
            )

            logger.info("✅ AnimateDiff loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load AnimateDiff: {e}")
            raise

    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("AnimateDiff unloaded")

    def generate_video_from_poses(
        self,
        source_image: Image.Image,
        pose_frames: List[np.ndarray],
        prompt: str = "a person dancing",
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        fps: int = 8,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate video from source image and pose sequence

        Args:
            source_image: Input photo (PIL Image)
            pose_frames: List of pose condition images (numpy arrays)
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            fps: Output video FPS
            output_path: Optional path to save video

        Returns:
            Dict with video frames and metadata
        """
        self.load_model()

        logger.info(f"🎬 Generating video with AnimateDiff")
        logger.info(f"   Frames: {len(pose_frames)}")
        logger.info(f"   Prompt: {prompt}")

        try:
            # Convert pose frames to control images
            control_images = []
            for pose_frame in pose_frames:
                # Convert to PIL Image
                if isinstance(pose_frame, np.ndarray):
                    pose_img = Image.fromarray(pose_frame)
                else:
                    pose_img = pose_frame
                control_images.append(pose_img)

            # Limit frames for memory
            max_frames = min(len(control_images), 16)
            control_images = control_images[:max_frames]

            logger.info(f"Processing {max_frames} frames")

            # Generate video
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=source_image,
                control_image=control_images,
                num_frames=max_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(42),
            )

            frames = output.frames[0]

            # Save video if path provided
            if output_path:
                export_to_video(frames, output_path, fps=fps)
                logger.info(f"✅ Video saved: {output_path}")

            return {
                "success": True,
                "frames": frames,
                "num_frames": len(frames),
                "output_path": output_path,
            }

        except Exception as e:
            logger.error(f"❌ AnimateDiff generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def create_pose_control_image(
        self,
        pose_landmarks: List[Dict],
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Create OpenPose-style control image from landmarks

        Args:
            pose_landmarks: List of pose landmarks
            width: Image width
            height: Image height

        Returns:
            Numpy array (RGB) of pose visualization
        """
        # Create black canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Define body connections
        connections = [
            (11, 12), (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]

                if start.get("visibility", 0) > 0.5 and end.get("visibility", 0) > 0.5:
                    start_point = (
                        int(start["x"] * width),
                        int(start["y"] * height)
                    )
                    end_point = (
                        int(end["x"] * width),
                        int(end["y"] * height)
                    )
                    cv2.line(canvas, start_point, end_point, (0, 255, 0), 3)

        # Draw keypoints
        for landmark in pose_landmarks:
            if landmark.get("visibility", 0) > 0.5:
                point = (
                    int(landmark["x"] * width),
                    int(landmark["y"] * height)
                )
                cv2.circle(canvas, point, 4, (0, 0, 255), -1)

        return canvas


# Singleton instance
_animatediff_service: Optional[AnimateDiffService] = None


def get_animatediff_service() -> AnimateDiffService:
    """Get singleton AnimateDiff service instance"""
    global _animatediff_service
    if _animatediff_service is None:
        _animatediff_service = AnimateDiffService()
    return _animatediff_service
