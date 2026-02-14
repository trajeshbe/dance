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

            # Step 3: Download input photo from MinIO
            update_progress("Processing input photo", 40)
            bucket, object_name = self.minio_client.parse_minio_url(photo_url)
            local_photo_path = os.path.join(self.work_dir, f"input_{object_name}")
            self.minio_client.client.fget_object(bucket, object_name, local_photo_path)

            # Step 4: Detect face in photo (for later face animation)
            update_progress("Detecting face in photo", 45)
            photo = Image.open(local_photo_path)
            photo_np = np.array(photo)

            # Step 5: Generate video with pose-guided motion
            update_progress("Generating body motion from poses", 50)
            # This is where we'd integrate AnimateDiff with ControlNet
            # For now, create a simple video by repeating the image with pose overlays
            output_frames = self._generate_pose_guided_frames(
                photo_np,
                pose_data,
                video_data["fps"],
            )

            # Step 6: Add facial expressions (if enabled)
            if enable_facial_expressions and pose_data["metadata"]["frames_with_face"] > 0:
                update_progress("Generating facial expressions", 70)
                # This is where we'd use FOMM or LivePortrait
                # For now, keep frames as-is
                pass

            # Step 7: Generate/replace background (if requested)
            if background_mode == "generated" and scene_prompt:
                update_progress(f"Generating background: {scene_prompt}", 80)
                # This is where we'd use background generation
                pass

            # Step 8: Composite final video
            update_progress("Compositing final video", 90)
            self._save_video(output_frames, output_path, video_data["fps"])

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
                }
            }

        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _generate_pose_guided_frames(
        self,
        photo: np.ndarray,
        pose_data: Dict[str, Any],
        fps: int,
    ) -> list:
        """
        Generate video frames with pose guidance

        For now, this creates a simple animated version by:
        1. Drawing pose skeleton over the photo
        2. Creating smooth transitions

        In production, this would use AnimateDiff with pose ControlNet
        """
        frames = []
        photo_rgb = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)

        for pose_frame in pose_data["poses"]:
            frame = photo_rgb.copy()

            # Draw pose skeleton if available
            if pose_frame["pose_landmarks"]:
                frame = self._draw_pose_skeleton(frame, pose_frame["pose_landmarks"])

            frames.append(frame)

        logger.info(f"Generated {len(frames)} frames")
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
