"""
Text-to-Video Generation Service (Sora/Kling-style)

Supports:
- Stable Video Diffusion (SVD)
- AnimateDiff
- Scene/style prompts for background generation
- Cinematic quality output
"""
import torch
import numpy as np
from typing import List, Optional, Dict
from PIL import Image
import cv2


class StableVideoDiffusionService:
    """
    Stable Video Diffusion for high-quality video generation

    Can generate:
    - Video from image + text prompt
    - Background scenes based on prompts
    - Cinematic style transfers
    """

    def __init__(
        self,
        model_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda"
    ):
        self.device = device
        self.model_path = model_path
        self._init_model()

    def _init_model(self):
        """Initialize Stable Video Diffusion model"""
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video

            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            self.pipe.to(self.device)
            self.pipe.enable_model_cpu_offload()  # Memory optimization

        except ImportError:
            print("WARNING: diffusers library not found")
            print("Install with: pip install diffusers transformers accelerate")
            raise

    def generate_background_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 120,  # ~4 seconds at 30fps
        fps: int = 30,
        width: int = 1024,
        height: int = 1024,
        motion_bucket_id: int = 127,  # Controls motion amount
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate background video from text prompt

        Args:
            prompt: Scene description (e.g., "neon-lit nightclub with dancing lights")
            negative_prompt: What to avoid
            num_frames: Number of frames to generate
            fps: Frames per second
            width, height: Output resolution
            motion_bucket_id: Amount of motion (0-255, higher = more motion)
            seed: Random seed for reproducibility

        Returns:
            List of video frames as numpy arrays (H, W, 3) RGB
        """
        # First, generate a static image from prompt using Stable Diffusion
        # Then use SVD to animate it
        from diffusers import StableDiffusionPipeline

        # Generate initial frame
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to(self.device)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        initial_image = sd_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=30,
            generator=generator
        ).images[0]

        # Generate video from initial image
        frames = self.pipe(
            initial_image,
            height=height,
            width=width,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            decode_chunk_size=8,  # Memory optimization
            generator=generator
        ).frames[0]

        # Convert PIL images to numpy arrays
        frames_np = [np.array(frame) for frame in frames]

        return frames_np

    def animate_from_image(
        self,
        image: np.ndarray,
        num_frames: int = 120,
        motion_bucket_id: int = 127,
        fps: int = 30,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Animate a static image (add subtle motion)

        Useful for making backgrounds more dynamic
        """
        # Convert numpy to PIL
        pil_image = Image.fromarray(image)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        frames = self.pipe(
            pil_image,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            generator=generator
        ).frames[0]

        frames_np = [np.array(frame) for frame in frames]
        return frames_np


class AnimateDiffService:
    """
    AnimateDiff for motion-controlled video generation

    Better for:
    - Controlled motion paths
    - Combining with ControlNet (pose, depth, etc.)
    - Dance-specific motion
    """

    def __init__(
        self,
        model_path: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        base_model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda"
    ):
        self.device = device
        self.model_path = model_path
        self.base_model = base_model
        self._init_model()

    def _init_model(self):
        """Initialize AnimateDiff pipeline"""
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

            # Load motion adapter
            adapter = MotionAdapter.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            )

            # Load pipeline
            self.pipe = AnimateDiffPipeline.from_pretrained(
                self.base_model,
                motion_adapter=adapter,
                torch_dtype=torch.float16
            ).to(self.device)

            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config,
                beta_schedule="linear",
                clip_sample=False
            )

        except ImportError:
            print("WARNING: AnimateDiff not available")
            raise

    def generate_with_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 16,  # AnimateDiff default
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate video from text prompt

        Args:
            prompt: What to generate (e.g., "person dancing energetically")
            negative_prompt: What to avoid
            num_frames: Number of frames
            guidance_scale: How strictly to follow prompt
            num_inference_steps: Quality vs speed tradeoff

        Returns:
            List of frames
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )

        frames = output.frames[0]
        frames_np = [np.array(frame) for frame in frames]

        return frames_np

    def generate_with_pose_control(
        self,
        prompt: str,
        pose_images: List[np.ndarray],  # Pose skeleton images
        negative_prompt: Optional[str] = None,
        controlnet_conditioning_scale: float = 1.0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate video with pose control (ControlNet + AnimateDiff)

        This is the BEST approach for dance videos with prompts!

        Args:
            prompt: Scene/style description
            pose_images: List of pose skeleton images (from OpenPose)
            negative_prompt: What to avoid
            controlnet_conditioning_scale: How strongly to follow pose
            guidance_scale: How strictly to follow text prompt
            num_inference_steps: Quality
            seed: Random seed

        Returns:
            Generated video frames
        """
        try:
            from diffusers import ControlNetModel

            # Load ControlNet for pose
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose",
                torch_dtype=torch.float16
            ).to(self.device)

            # Recreate pipeline with ControlNet
            from diffusers import AnimateDiffControlNetPipeline

            pipe = AnimateDiffControlNetPipeline.from_pretrained(
                self.base_model,
                controlnet=controlnet,
                torch_dtype=torch.float16
            ).to(self.device)

            # Convert pose images to PIL
            pose_pil = [Image.fromarray(img) for img in pose_images]

            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator = generator.manual_seed(seed)

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pose_pil,
                num_frames=len(pose_images),
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            )

            frames = output.frames[0]
            frames_np = [np.array(frame) for frame in frames]

            return frames_np

        except Exception as e:
            print(f"ControlNet generation failed: {e}")
            # Fallback to regular generation
            return self.generate_with_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=len(pose_images),
                seed=seed
            )


class BackgroundGenerationService:
    """
    Service for generating/replacing backgrounds in dance videos

    Combines:
    - Text-to-image for static backgrounds
    - SVD for animated backgrounds
    - Segmentation for person extraction
    """

    def __init__(self):
        self.svd_service = None  # Lazy load
        self.animatediff_service = None  # Lazy load

    def generate_static_background(
        self,
        prompt: str,
        style_prompt: str = "cinematic, 4k, professional lighting",
        negative_prompt: str = "low quality, blurry, distorted",
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate static background from prompt

        Args:
            prompt: Scene description ("neon nightclub", "beach sunset")
            style_prompt: Quality/style modifiers
            negative_prompt: What to avoid
            width, height: Resolution
            seed: Random seed

        Returns:
            Background image (H, W, 3) RGB
        """
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")

        full_prompt = f"{prompt}, {style_prompt}"

        generator = torch.Generator(device="cuda")
        if seed is not None:
            generator = generator.manual_seed(seed)

        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=30,
            generator=generator
        ).images[0]

        return np.array(image)

    def generate_animated_background(
        self,
        prompt: str,
        num_frames: int = 120,
        fps: int = 30,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate animated background video

        Args:
            prompt: Scene description
            num_frames: Number of frames
            fps: Frame rate
            seed: Random seed

        Returns:
            List of background frames
        """
        if self.svd_service is None:
            self.svd_service = StableVideoDiffusionService()

        return self.svd_service.generate_background_video(
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            seed=seed
        )

    def composite_person_on_background(
        self,
        person_frames: List[np.ndarray],
        background_frames: List[np.ndarray],
        person_masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Composite person onto generated background

        Args:
            person_frames: Frames with person (may have original background)
            background_frames: Generated background frames
            person_masks: Segmentation masks for person

        Returns:
            Composited frames
        """
        composited = []

        for person, background, mask in zip(person_frames, background_frames, person_masks):
            # Resize background to match person frame
            if background.shape[:2] != person.shape[:2]:
                background = cv2.resize(
                    background,
                    (person.shape[1], person.shape[0])
                )

            # Ensure mask is 3-channel
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]
            if mask.shape[2] == 1:
                mask = np.repeat(mask, 3, axis=2)

            # Normalize mask to 0-1
            mask = mask.astype(float) / 255.0

            # Composite: person * mask + background * (1 - mask)
            composited_frame = (
                person.astype(float) * mask +
                background.astype(float) * (1 - mask)
            ).astype(np.uint8)

            composited.append(composited_frame)

        return composited


# Example usage for Sora/Kling-style generation
def generate_sora_style_dance_video(
    person_image: np.ndarray,
    pose_sequence: List[np.ndarray],
    scene_prompt: str,
    style_prompt: str = "cinematic, 4k, dramatic lighting",
    background_mode: str = "generated"  # "original", "generated", "custom"
) -> List[np.ndarray]:
    """
    Generate Sora/Kling quality dance video with prompts

    Args:
        person_image: Source person image
        pose_sequence: List of pose skeleton images
        scene_prompt: "dancing in a neon nightclub"
        style_prompt: "cinematic, 4k, professional"
        background_mode: How to handle background

    Returns:
        Final video frames
    """
    # Initialize services
    animatediff = AnimateDiffService()

    # Build full prompt
    full_prompt = f"{scene_prompt}, {style_prompt}, person dancing"

    # Generate with pose control
    frames = animatediff.generate_with_pose_control(
        prompt=full_prompt,
        pose_images=pose_sequence,
        negative_prompt="low quality, blurry, distorted, deformed",
        controlnet_conditioning_scale=1.0,
        guidance_scale=7.5
    )

    return frames
