"""
Open-Sora 2.0 Integration

Open-Sora is an open-source text-to-video model that rivals OpenAI's Sora.

Model: hpcaitech/Open-Sora
GitHub: https://github.com/hpcaitech/Open-Sora
Paper: https://arxiv.org/abs/2410.15953

Features:
- Text-to-Video generation (up to 1024x1024, 16s)
- Image-to-Video
- Video-to-Video
- High quality, cinematic output
- 11B parameter model (Open-Sora 2.0)

Perfect for generating dance videos from text prompts!
"""
import torch
from typing import List, Optional
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class OpenSoraService:
    """
    Open-Sora 2.0 for text-to-video generation

    Superior to Stable Video Diffusion for:
    - Longer videos (up to 16 seconds)
    - Better text understanding
    - More cinematic quality
    - Dance-specific motion
    """

    def __init__(
        self,
        model_path: str = "hpcaitech/Open-Sora-Plan-v1.3.0",
        device: str = "cuda"
    ):
        self.device = device
        self.model_path = model_path
        self._init_model()

    def _init_model(self):
        """Initialize Open-Sora model"""
        try:
            # TODO: Install Open-Sora
            # pip install git+https://github.com/hpcaitech/Open-Sora.git

            from opensora.models import OpenSoraPipeline

            self.pipe = OpenSoraPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            ).to(self.device)

            logger.info(f"Open-Sora loaded on {self.device}")

        except ImportError:
            logger.warning("Open-Sora not installed. Install with:")
            logger.warning("git clone https://github.com/hpcaitech/Open-Sora.git")
            logger.warning("cd Open-Sora && pip install -e .")
            raise

    def generate_dance_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 120,  # 4 seconds at 30fps
        height: int = 1024,
        width: int = 1024,
        fps: int = 30,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate dance video from text prompt using Open-Sora

        Args:
            prompt: Text description
                    Example: "A person dancing energetically in a neon nightclub,
                             cinematic lighting, high quality, 4k"
            negative_prompt: What to avoid
            num_frames: Number of frames (max ~480 for 16s)
            height, width: Resolution (512, 720, 1024)
            fps: Frame rate
            guidance_scale: How strictly to follow prompt
            num_inference_steps: Quality vs speed
            seed: Random seed

        Returns:
            List of video frames (H, W, 3) RGB
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        # Generate video
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "low quality, blurry, distorted",
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        # Convert to numpy
        frames = output.frames[0]
        frames_np = [np.array(frame) for frame in frames]

        return frames_np

    def image_to_video(
        self,
        image: np.ndarray,
        prompt: str,
        num_frames: int = 120,
        motion_strength: float = 1.0,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate video from starting image + text prompt

        Useful for animating a person's photo with dance motion

        Args:
            image: Starting frame (H, W, 3) RGB
            prompt: Motion description
                    Example: "person dancing to energetic music"
            num_frames: Number of frames
            motion_strength: Amount of motion (0-2)
            seed: Random seed

        Returns:
            List of video frames
        """
        pil_image = Image.fromarray(image)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            image=pil_image,
            num_frames=num_frames,
            strength=motion_strength,
            generator=generator
        )

        frames = output.frames[0]
        frames_np = [np.array(frame) for frame in frames]

        return frames_np

    def generate_with_pose_control(
        self,
        prompt: str,
        pose_sequence: List[np.ndarray],
        person_image: np.ndarray,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate dance video with pose control + person image

        This combines:
        - Text prompt for scene/style
        - Pose sequence for motion
        - Person image for identity

        Best approach for our dance generator!

        Args:
            prompt: Scene/style description
            pose_sequence: List of pose images (OpenPose skeletons)
            person_image: Person to dance
            seed: Random seed

        Returns:
            Video frames with person dancing in scene
        """
        # Convert person image to PIL
        person_pil = Image.fromarray(person_image)

        # Convert pose sequence to PIL
        pose_pil = [Image.fromarray(pose) for pose in pose_sequence]

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        # Generate with pose + image conditioning
        output = self.pipe(
            prompt=prompt,
            image=person_pil,
            control_images=pose_pil,  # Pose control
            num_frames=len(pose_sequence),
            controlnet_conditioning_scale=1.0,
            generator=generator
        )

        frames = output.frames[0]
        frames_np = [np.array(frame) for frame in frames]

        return frames_np


# Example usage for dance video generation
def generate_dance_with_opensora(
    person_image: np.ndarray,
    pose_sequence: List[np.ndarray],
    scene_prompt: str = "dancing in a neon nightclub",
    style_prompt: str = "cinematic, 4k, dramatic lighting",
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate dance video using Open-Sora

    This is the BEST approach for our use case:
    - Better than AnimateDiff (longer videos, better quality)
    - Better than Stable Video Diffusion (better text control)
    - Open-source alternative to Sora/Kling

    Args:
        person_image: Source person photo
        pose_sequence: Dance motion (from reference video)
        scene_prompt: Scene description
        style_prompt: Quality modifiers
        seed: Random seed

    Returns:
        Final dance video frames
    """
    service = OpenSoraService()

    # Build full prompt
    full_prompt = f"A person {scene_prompt}, {style_prompt}, professional video"

    # Generate with pose control
    frames = service.generate_with_pose_control(
        prompt=full_prompt,
        pose_sequence=pose_sequence,
        person_image=person_image,
        seed=seed
    )

    return frames
