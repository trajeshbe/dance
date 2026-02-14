"""
Video Compositor Service

Handles:
- Combining body motion and facial expression videos
- Background replacement/generation
- Multi-person composition
- Final rendering
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoCompositor:
    """Service for compositing final videos"""

    def __init__(self):
        """Initialize compositor"""
        logger.info("🎬 Initializing Video Compositor")

    def composite_face_on_body(
        self,
        body_video_path: str,
        face_video_path: str,
        output_path: str,
        blend_strength: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Composite facial expression video onto body motion video

        Args:
            body_video_path: Path to body motion video
            face_video_path: Path to facial expression video
            output_path: Path for output video
            blend_strength: Blending strength (0-1)

        Returns:
            Dict with success and output_path
        """
        try:
            logger.info("🎭 Compositing face onto body")

            # Open videos
            body_cap = cv2.VideoCapture(body_video_path)
            face_cap = cv2.VideoCapture(face_video_path)

            if not body_cap.isOpened() or not face_cap.isOpened():
                raise ValueError("Could not open video files")

            # Get video properties
            fps = int(body_cap.get(cv2.CAP_PROP_FPS))
            width = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret_body, body_frame = body_cap.read()
                ret_face, face_frame = face_cap.read()

                if not ret_body or not ret_face:
                    break

                # Resize face to match body
                if face_frame.shape[:2] != body_frame.shape[:2]:
                    face_frame = cv2.resize(face_frame, (width, height))

                # Simple alpha blending
                # In production, this would use face detection and precise masking
                composited = cv2.addWeighted(
                    body_frame, 1.0 - blend_strength,
                    face_frame, blend_strength,
                    0
                )

                out.write(composited)
                frame_count += 1

                if frame_count % 30 == 0:
                    logger.info(f"Composited {frame_count} frames")

            body_cap.release()
            face_cap.release()
            out.release()

            logger.info(f"✅ Composited {frame_count} frames: {output_path}")

            return {
                "success": True,
                "output_path": output_path,
                "num_frames": frame_count,
            }

        except Exception as e:
            logger.error(f"❌ Compositing failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def replace_background(
        self,
        video_path: str,
        background_image: Image.Image,
        output_path: str,
        use_green_screen: bool = False,
    ) -> Dict[str, Any]:
        """
        Replace video background

        Args:
            video_path: Path to input video
            background_image: PIL Image for background
            output_path: Path for output video
            use_green_screen: Whether to use chroma key

        Returns:
            Dict with success and output_path
        """
        try:
            logger.info("🌄 Replacing background")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Resize background
            bg_resized = background_image.resize((width, height))
            bg_array = np.array(bg_resized)
            bg_bgr = cv2.cvtColor(bg_array, cv2.COLOR_RGB2BGR)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if use_green_screen:
                    # Simple green screen keying
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([40, 40, 40])
                    upper_green = np.array([80, 255, 255])
                    mask = cv2.inRange(hsv, lower_green, upper_green)
                    mask_inv = cv2.bitwise_not(mask)

                    # Extract foreground
                    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)

                    # Extract background
                    bg = cv2.bitwise_and(bg_bgr, bg_bgr, mask=mask)

                    # Combine
                    composited = cv2.add(fg, bg)
                else:
                    # Simple blending (for demo)
                    composited = cv2.addWeighted(frame, 0.7, bg_bgr, 0.3, 0)

                out.write(composited)
                frame_count += 1

            cap.release()
            out.release()

            logger.info(f"✅ Background replaced: {output_path}")

            return {
                "success": True,
                "output_path": output_path,
                "num_frames": frame_count,
            }

        except Exception as e:
            logger.error(f"❌ Background replacement failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def concatenate_videos(
        self,
        video_paths: List[str],
        output_path: str,
        layout: str = "grid",  # "grid", "horizontal", "vertical"
    ) -> Dict[str, Any]:
        """
        Concatenate multiple videos

        Args:
            video_paths: List of video paths
            output_path: Output path
            layout: Layout type

        Returns:
            Dict with success and output_path
        """
        try:
            logger.info(f"🎞️ Concatenating {len(video_paths)} videos ({layout} layout)")

            if not video_paths:
                raise ValueError("No videos to concatenate")

            # Open all videos
            caps = [cv2.VideoCapture(path) for path in video_paths]

            if not all(cap.isOpened() for cap in caps):
                raise ValueError("Could not open all video files")

            # Get properties from first video
            fps = int(caps[0].get(cv2.CAP_PROP_FPS))
            width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate output dimensions based on layout
            if layout == "horizontal":
                out_width = width * len(video_paths)
                out_height = height
            elif layout == "vertical":
                out_width = width
                out_height = height * len(video_paths)
            elif layout == "grid":
                cols = int(np.ceil(np.sqrt(len(video_paths))))
                rows = int(np.ceil(len(video_paths) / cols))
                out_width = width * cols
                out_height = height * rows
            else:
                raise ValueError(f"Unknown layout: {layout}")

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

            frame_count = 0

            while True:
                frames = []
                all_read = True

                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        all_read = False
                        break
                    frames.append(frame)

                if not all_read:
                    break

                # Compose based on layout
                if layout == "horizontal":
                    composited = np.hstack(frames)
                elif layout == "vertical":
                    composited = np.vstack(frames)
                elif layout == "grid":
                    # Arrange in grid
                    grid_frames = frames + [np.zeros_like(frames[0])] * (cols * rows - len(frames))
                    rows_list = []
                    for r in range(rows):
                        row_frames = grid_frames[r * cols:(r + 1) * cols]
                        rows_list.append(np.hstack(row_frames))
                    composited = np.vstack(rows_list)

                out.write(composited)
                frame_count += 1

            for cap in caps:
                cap.release()
            out.release()

            logger.info(f"✅ Concatenated video: {output_path}")

            return {
                "success": True,
                "output_path": output_path,
                "num_frames": frame_count,
            }

        except Exception as e:
            logger.error(f"❌ Concatenation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def add_effects(
        self,
        video_path: str,
        output_path: str,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Add color grading effects to video

        Args:
            video_path: Input video path
            output_path: Output video path
            brightness: Brightness multiplier
            contrast: Contrast multiplier
            saturation: Saturation multiplier

        Returns:
            Dict with success and output_path
        """
        try:
            logger.info("✨ Adding effects to video")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to HSV for saturation adjustment
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

                # Adjust saturation
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)

                # Convert back to BGR
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # Adjust brightness and contrast
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness * 30)

                out.write(frame)
                frame_count += 1

            cap.release()
            out.release()

            logger.info(f"✅ Effects added: {output_path}")

            return {
                "success": True,
                "output_path": output_path,
                "num_frames": frame_count,
            }

        except Exception as e:
            logger.error(f"❌ Adding effects failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Singleton instance
_compositor: Optional[VideoCompositor] = None


def get_video_compositor() -> VideoCompositor:
    """Get singleton video compositor instance"""
    global _compositor
    if _compositor is None:
        _compositor = VideoCompositor()
    return _compositor
