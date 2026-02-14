"""
Dance Video Generation Pipeline

Orchestrates the full dance video generation workflow:
1. Download reference video
2. Extract poses and facial landmarks
3. Process input photo
4. Generate body motion with poses
5. Generate facial expressions
6. Composite final video
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from app.services.youtube_service import get_youtube_service
from app.services.pose_extraction_service import get_pose_extraction_service
from app.services.minio_client import get_minio_client
from app.services.models.animatediff_service import get_animatediff_service
from app.services.audio_service import get_audio_service
from app.services.video_compositor import get_video_compositor

logger = logging.getLogger(__name__)


class DanceGenerationPipeline:
    """Full pipeline for generating dance videos"""

    def __init__(self, work_dir: str = "/tmp/dance_generation"):
        """
        Initialize pipeline

        Args:
            work_dir: Working directory for temporary files
        """
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        self.youtube_service = get_youtube_service()
        self.pose_service = get_pose_extraction_service()
        self.minio_client = get_minio_client()
        self.animatediff_service = get_animatediff_service()
        self.audio_service = get_audio_service()
        self.compositor = get_video_compositor()

    async def generate_dance_video(
        self,
        photo_url: str,
        reference_video_url: str,
        output_path: str,
        scene_prompt: Optional[str] = None,
        style_prompt: str = "cinematic, 4k, dramatic lighting",
        background_mode: str = "original",
        enable_facial_expressions: bool = True,
        enable_lip_sync: bool = True,
        expression_intensity: float = 1.0,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate dance video from photo and reference

        Args:
            photo_url: MinIO URL to input photo
            reference_video_url: YouTube URL of reference dance
            output_path: Path to save output video
            scene_prompt: Optional scene description
            style_prompt: Style modifiers
            background_mode: 'original' or 'generated'
            enable_facial_expressions: Whether to animate face
            enable_lip_sync: Whether to sync lips to audio
            expression_intensity: Expression strength (0-2)
            progress_callback: Callback for progress updates

        Returns:
            Dict with output path and metadata
        """
        try:
            def update_progress(step: str, progress: int):
                """Helper to update progress"""
                logger.info(f"[{progress}%] {step}")
                if progress_callback:
                    progress_callback(step, progress)

            update_progress("Starting video generation", 0)

            # Step 1: Download reference video
            update_progress("Downloading reference video from YouTube", 10)
            reference_id = os.path.basename(reference_video_url).replace("watch?v=", "")[:16]
            video_data = self.youtube_service.download_video(
                reference_video_url,
                video_id=reference_id,
            )
            reference_video_path = video_data["video_path"]

            # Step 2: Extract poses from reference
            update_progress("Extracting poses and facial landmarks", 25)
            pose_data = self.pose_service.extract_poses_from_video(
                reference_video_path,
                extract_face=enable_facial_expressions,
                extract_hands=False,
                max_frames=150,  # Limit to ~5 seconds at 30fps
            )

            # Save pose data
            pose_json_path = os.path.join(self.work_dir, f"{reference_id}_poses.json")
            self.pose_service.save_poses_to_file(pose_data, pose_json_path)

            # Step 3: Extract audio from reference video
            update_progress("Extracting audio from reference video", 35)
            audio_result = self.audio_service.extract_audio_from_video(reference_video_path)
            audio_path = audio_result.get("audio_path") if audio_result["success"] else None

            # Step 3.5: Detect beats for synchronization (if audio available)
            beat_data = None
            if audio_path and enable_lip_sync:
                update_progress("Detecting beats for synchronization", 38)
                beat_result = self.audio_service.detect_beats(audio_path)
                if beat_result["success"]:
                    beat_data = beat_result
                    logger.info(f"Detected {beat_data['num_beats']} beats at {beat_data['tempo']:.1f} BPM")

            # Step 4: Download input photo from MinIO
            update_progress("Processing input photo", 40)
            bucket, object_name = self.minio_client.parse_minio_url(photo_url)
            local_photo_path = os.path.join(self.work_dir, f"input_{object_name}")
            self.minio_client.client.fget_object(bucket, object_name, local_photo_path)

            # Step 5: Load photo
            update_progress("Loading input photo", 45)
            photo = Image.open(local_photo_path).convert("RGB")
            photo_np = np.array(photo)

            # Step 6: Generate video with pose-guided motion using AnimateDiff
            update_progress("Generating body motion with AI (AnimateDiff + ControlNet)", 50)
            output_frames = self._generate_pose_guided_frames_ai(
                photo,
                pose_data,
                video_data["fps"],
                scene_prompt=scene_prompt,
                style_prompt=style_prompt,
            )

            # Step 7: Save body motion video (temporary)
            update_progress("Saving body motion video", 70)
            temp_body_video = os.path.join(self.work_dir, f"{reference_id}_body.mp4")
            self._save_video(output_frames, temp_body_video, video_data["fps"])

            # Step 8: Add facial expressions (if enabled)
            final_video_path = temp_body_video
            if enable_facial_expressions and pose_data["metadata"]["frames_with_face"] > 0:
                update_progress("Generating facial expressions (placeholder)", 75)
                # TODO: Integrate LivePortrait or FOMM for facial animation
                # For now, skip this step - body motion is already animated
                logger.info("Facial expression generation not yet implemented - using body motion only")

            # Step 9: Apply color grading effects (if requested)
            if style_prompt and ("vibrant" in style_prompt.lower() or "cinematic" in style_prompt.lower()):
                update_progress("Applying cinematic color grading", 80)
                temp_graded_video = os.path.join(self.work_dir, f"{reference_id}_graded.mp4")

                # Apply effects based on style
                saturation = 1.2 if "vibrant" in style_prompt.lower() else 1.0
                brightness = 1.1 if "bright" in style_prompt.lower() else 1.0
                contrast = 1.15 if "cinematic" in style_prompt.lower() else 1.0

                effect_result = self.compositor.add_effects(
                    final_video_path,
                    temp_graded_video,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                )

                if effect_result["success"]:
                    final_video_path = temp_graded_video

            # Step 10: Add audio to final video
            update_progress("Adding synchronized audio", 90)
            if audio_path:
                temp_with_audio = os.path.join(self.work_dir, f"{reference_id}_with_audio.mp4")
                audio_result = self.audio_service.add_audio_to_video(
                    final_video_path,
                    audio_path,
                    temp_with_audio,
                    sync_offset=0.0,  # TODO: Calculate sync offset from beat data
                )

                if audio_result["success"]:
                    final_video_path = temp_with_audio
                    logger.info("✅ Audio synchronized to video")

            # Step 11: Copy to final output path
            update_progress("Finalizing video", 95)
            import shutil
            shutil.copy(final_video_path, output_path)

            # Step 9: Upload to MinIO
            update_progress("Uploading final video", 95)
            output_filename = os.path.basename(output_path)
            with open(output_path, 'rb') as f:
                video_content = f.read()
                from io import BytesIO
                self.minio_client.client.put_object(
                    "dance-videos",
                    output_filename,
                    BytesIO(video_content),
                    length=len(video_content),
                    content_type="video/mp4"
                )

            update_progress("Video generation complete!", 100)

            return {
                "success": True,
                "output_path": output_path,
                "video_url": f"minio://dance-videos/{output_filename}",
                "metadata": {
                    "reference_video": reference_video_url,
                    "num_frames": len(output_frames),
                    "fps": video_data["fps"],
                    "duration": len(output_frames) / video_data["fps"],
                    "poses_extracted": pose_data["metadata"]["frames_with_pose"],
                    "face_detected": pose_data["metadata"]["frames_with_face"] > 0,
                    "has_audio": audio_path is not None,
                    "audio_tempo": beat_data["tempo"] if beat_data else None,
                    "num_beats": beat_data["num_beats"] if beat_data else None,
                    "scene_prompt": scene_prompt,
                    "style_prompt": style_prompt,
                }
            }

        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _generate_pose_guided_frames_ai(
        self,
        photo: Image.Image,
        pose_data: Dict[str, Any],
        fps: int,
        scene_prompt: Optional[str] = None,
        style_prompt: str = "cinematic, 4k, dramatic lighting",
    ) -> list:
        """
        Generate video frames using AnimateDiff with pose ControlNet

        Args:
            photo: Input photo (PIL Image)
            pose_data: Extracted pose data from reference video
            fps: Target FPS
            scene_prompt: Optional scene description
            style_prompt: Style modifiers

        Returns:
            List of video frames (numpy arrays in BGR format)
        """
        try:
            logger.info("🎬 Generating video with AnimateDiff + ControlNet")

            # Get image dimensions
            width, height = photo.size

            # Create pose control images from landmarks
            pose_frames = []
            for pose_frame in pose_data["poses"]:
                if pose_frame["pose_landmarks"]:
                    control_image = self.animatediff_service.create_pose_control_image(
                        pose_frame["pose_landmarks"],
                        width,
                        height,
                    )
                    pose_frames.append(control_image)

            if not pose_frames:
                logger.warning("No pose data found, falling back to simple animation")
                return self._generate_pose_guided_frames_fallback(np.array(photo), pose_data, fps)

            # Construct full prompt
            if scene_prompt:
                full_prompt = f"a person {scene_prompt}, {style_prompt}"
            else:
                full_prompt = f"a person dancing, {style_prompt}"

            logger.info(f"Prompt: {full_prompt}")
            logger.info(f"Pose frames: {len(pose_frames)}")

            # Generate video with AnimateDiff
            result = self.animatediff_service.generate_video_from_poses(
                source_image=photo,
                pose_frames=pose_frames,
                prompt=full_prompt,
                negative_prompt="blurry, low quality, distorted, deformed, static, duplicate, multiple people",
                num_inference_steps=25,  # Balance quality vs speed
                guidance_scale=7.5,
                fps=fps,
            )

            if not result["success"]:
                logger.error(f"AnimateDiff failed: {result.get('error')}")
                logger.info("Falling back to simple pose overlay")
                return self._generate_pose_guided_frames_fallback(np.array(photo), pose_data, fps)

            # Convert PIL frames to numpy BGR format for OpenCV
            frames = []
            for pil_frame in result["frames"]:
                # Convert PIL Image to numpy array
                if isinstance(pil_frame, Image.Image):
                    frame_np = np.array(pil_frame)
                else:
                    frame_np = pil_frame

                # Convert RGB to BGR for OpenCV
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_np

                frames.append(frame_bgr)

            logger.info(f"✅ Generated {len(frames)} AI-powered frames")
            return frames

        except Exception as e:
            logger.error(f"❌ AI generation failed: {e}", exc_info=True)
            logger.info("Falling back to simple pose overlay")
            return self._generate_pose_guided_frames_fallback(np.array(photo), pose_data, fps)

    def _generate_pose_guided_frames_fallback(
        self,
        photo: np.ndarray,
        pose_data: Dict[str, Any],
        fps: int,
    ) -> list:
        """
        Fallback: Generate simple video frames with pose skeleton overlay

        This creates a basic animated version by:
        1. Drawing pose skeleton over the photo
        2. Creating frame-by-frame visualization

        Used when AI models fail or aren't available
        """
        frames = []
        photo_bgr = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR) if len(photo.shape) == 3 else photo

        for pose_frame in pose_data["poses"]:
            frame = photo_bgr.copy()

            # Draw pose skeleton if available
            if pose_frame["pose_landmarks"]:
                frame = self._draw_pose_skeleton(frame, pose_frame["pose_landmarks"])

            frames.append(frame)

        logger.info(f"Generated {len(frames)} fallback frames")
        return frames

    def _draw_pose_skeleton(self, frame: np.ndarray, landmarks: list) -> np.ndarray:
        """Draw pose skeleton on frame"""
        h, w = frame.shape[:2]

        # Define connections between landmarks (simplified)
        connections = [
            (11, 12), (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24), (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                if start["visibility"] > 0.5 and end["visibility"] > 0.5:
                    start_point = (int(start["x"] * w), int(start["y"] * h))
                    end_point = (int(end["x"] * w), int(end["y"] * h))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Draw keypoints
        for landmark in landmarks:
            if landmark["visibility"] > 0.5:
                point = (int(landmark["x"] * w), int(landmark["y"] * h))
                cv2.circle(frame, point, 3, (0, 0, 255), -1)

        return frame

    def _save_video(self, frames: list, output_path: str, fps: int):
        """Save frames as video"""
        if not frames:
            raise ValueError("No frames to save")

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()
        logger.info(f"💾 Saved video: {output_path}")


# Singleton instance
_pipeline: Optional[DanceGenerationPipeline] = None


def get_dance_generation_pipeline() -> DanceGenerationPipeline:
    """Get singleton pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = DanceGenerationPipeline()
    return _pipeline
