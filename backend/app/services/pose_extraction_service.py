"""
Pose Extraction Service

Extracts pose keypoints from videos using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Dict, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class PoseExtractionService:
    """Service for extracting pose keypoints from videos"""

    def __init__(self):
        """Initialize MediaPipe Pose"""
        try:
            # Try new MediaPipe API first
            if hasattr(mp, 'solutions'):
                self.mp_pose = mp.solutions.pose
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_hands = mp.solutions.hands
            else:
                # Fallback for older MediaPipe versions
                from mediapipe.python.solutions import pose, face_mesh, hands
                self.mp_pose = pose
                self.mp_face_mesh = face_mesh
                self.mp_hands = hands
            logger.info("✅ MediaPipe initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise

    def extract_poses_from_video(
        self,
        video_path: str,
        extract_face: bool = True,
        extract_hands: bool = True,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract pose keypoints from video

        Args:
            video_path: Path to video file
            extract_face: Whether to extract facial landmarks
            extract_hands: Whether to extract hand landmarks
            max_frames: Maximum number of frames to process

        Returns:
            Dict with pose data, face data, and metadata
        """
        logger.info(f"🎯 Extracting poses from: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")

        if max_frames:
            total_frames = min(total_frames, max_frames)

        # Initialize MediaPipe
        pose_results = []
        face_results = []

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            face_mesh = None
            if extract_face:
                face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )

            frame_count = 0
            processed_count = 0

            while cap.isOpened() and (max_frames is None or processed_count < max_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process every frame (or skip frames for speed)
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract pose
                pose_result = pose.process(rgb_frame)

                frame_data = {
                    "frame": frame_count,
                    "pose_landmarks": None,
                    "face_landmarks": None,
                }

                if pose_result.pose_landmarks:
                    # Convert landmarks to list of dicts
                    landmarks = []
                    for landmark in pose_result.pose_landmarks.landmark:
                        landmarks.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility,
                        })
                    frame_data["pose_landmarks"] = landmarks

                # Extract face landmarks
                if extract_face and face_mesh:
                    face_result = face_mesh.process(rgb_frame)
                    if face_result.multi_face_landmarks:
                        # Just store the first face for simplicity
                        face_landmarks = []
                        for landmark in face_result.multi_face_landmarks[0].landmark:
                            face_landmarks.append({
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                            })
                        frame_data["face_landmarks"] = face_landmarks

                pose_results.append(frame_data)
                processed_count += 1

                if processed_count % 30 == 0:
                    logger.info(f"Processed {processed_count}/{total_frames} frames")

            if face_mesh:
                face_mesh.close()

        cap.release()

        logger.info(f"✅ Extracted poses from {processed_count} frames")

        # Calculate statistics
        frames_with_pose = sum(1 for f in pose_results if f["pose_landmarks"] is not None)
        frames_with_face = sum(1 for f in pose_results if f["face_landmarks"] is not None)

        return {
            "poses": pose_results,
            "metadata": {
                "total_frames": processed_count,
                "frames_with_pose": frames_with_pose,
                "frames_with_face": frames_with_face,
                "fps": fps,
                "width": width,
                "height": height,
                "pose_detection_rate": frames_with_pose / processed_count if processed_count > 0 else 0,
            }
        }

    def save_poses_to_file(self, pose_data: Dict[str, Any], output_path: str):
        """Save pose data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(pose_data, f)
        logger.info(f"💾 Saved poses to: {output_path}")

    def load_poses_from_file(self, input_path: str) -> Dict[str, Any]:
        """Load pose data from JSON file"""
        with open(input_path, 'r') as f:
            return json.load(f)

    def detect_num_dancers(self, pose_data: Dict[str, Any]) -> int:
        """
        Estimate number of dancers from pose data

        Simple heuristic: count number of unique pose tracks
        """
        # For now, just return 1 if poses detected, otherwise 0
        # This is a simplified version - proper multi-person tracking would be more complex
        if pose_data["metadata"]["frames_with_pose"] > 0:
            return 1
        return 0


# Singleton instance
_pose_service: Optional[PoseExtractionService] = None


def get_pose_extraction_service() -> PoseExtractionService:
    """Get singleton pose extraction service instance"""
    global _pose_service
    if _pose_service is None:
        _pose_service = PoseExtractionService()
    return _pose_service
