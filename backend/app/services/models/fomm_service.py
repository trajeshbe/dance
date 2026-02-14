"""
First Order Motion Model (FOMM) Service
For facial expression transfer with TPS (Thin Plate Spline)
"""
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple
import yaml
from pathlib import Path


class FOMMService:
    """
    First Order Motion Model for facial expression transfer

    Paper: "First Order Motion Model for Image Animation" (2019)
    https://arxiv.org/abs/2003.00196

    Features:
    - Self-supervised keypoint detection
    - Dense motion fields using Thin Plate Spline (TPS)
    - No 3D model required
    - Works with any object class
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda"
    ):
        """
        Initialize FOMM model

        Args:
            checkpoint_path: Path to trained model (.pth file)
            config_path: Path to model config (.yaml file)
            device: "cuda" or "cpu"
        """
        self.device = device

        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load model (will implement after installing FOMM)
        self.checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize generator and keypoint detector
        self._init_models()

    def _init_models(self):
        """Initialize FOMM generator and keypoint detector"""
        # This requires the FOMM library to be installed
        try:
            from modules.generator import OcclusionAwareGenerator
            from modules.keypoint_detector import KPDetector

            # Generator
            generator_params = self.config['model_params']['generator_params']
            self.generator = OcclusionAwareGenerator(
                **generator_params,
                **self.config['model_params']['common_params']
            )
            self.generator.load_state_dict(self.checkpoint['generator'])
            self.generator.to(self.device)
            self.generator.eval()

            # Keypoint detector
            kp_detector_params = self.config['model_params']['kp_detector_params']
            self.kp_detector = KPDetector(
                **kp_detector_params,
                **self.config['model_params']['common_params']
            )
            self.kp_detector.load_state_dict(self.checkpoint['kp_detector'])
            self.kp_detector.to(self.device)
            self.kp_detector.eval()

        except ImportError:
            print("WARNING: FOMM modules not found. Install from source:")
            print("git clone https://github.com/AliaksandrSiarohin/first-order-model.git")
            print("cd first-order-model && pip install -e .")
            raise

    def animate_face(
        self,
        source_image: np.ndarray,
        driving_video: List[np.ndarray],
        relative: bool = True,
        adapt_movement_scale: bool = True
    ) -> List[np.ndarray]:
        """
        Animate source face with expressions from driving video

        Args:
            source_image: Source face image (H, W, 3) RGB, values 0-255
            driving_video: List of driving video frames (H, W, 3) RGB
            relative: Use relative keypoint movement (recommended)
            adapt_movement_scale: Adapt movement scale to source

        Returns:
            List of generated frames with transferred expressions
        """
        # Preprocess source image
        source = self._preprocess_image(source_image)
        source = source.to(self.device)

        # Extract source keypoints
        with torch.no_grad():
            kp_source = self.kp_detector(source)

        # Process each driving frame
        generated_frames = []

        for i, driving_frame in enumerate(driving_video):
            # Preprocess driving frame
            driving = self._preprocess_image(driving_frame)
            driving = driving.to(self.device)

            # Extract driving keypoints
            with torch.no_grad():
                kp_driving = self.kp_detector(driving)

            # Make keypoints relative to first frame
            if relative:
                if i == 0:
                    kp_driving_initial = kp_driving

                kp_driving_relative = self._relative_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    adapt_movement_scale=adapt_movement_scale
                )
            else:
                kp_driving_relative = kp_driving

            # Generate frame
            with torch.no_grad():
                out = self.generator(
                    source,
                    kp_source=kp_source,
                    kp_driving=kp_driving_relative
                )

            # Post-process and convert to numpy
            generated = self._postprocess_image(out['prediction'])
            generated_frames.append(generated)

        return generated_frames

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for FOMM

        Args:
            image: RGB image (H, W, 3), values 0-255

        Returns:
            Normalized tensor (1, 3, H, W), values -1 to 1
        """
        # Resize to 256x256 (FOMM default)
        image = cv2.resize(image, (256, 256))

        # Convert to float and normalize to [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        # Convert to tensor: (H, W, 3) -> (3, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return image

    def _postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor output to numpy image

        Args:
            tensor: (1, 3, H, W), values -1 to 1

        Returns:
            RGB image (H, W, 3), values 0-255
        """
        # Remove batch dimension and move to CPU
        image = tensor.squeeze(0).cpu().numpy()

        # (3, H, W) -> (H, W, 3)
        image = np.transpose(image, (1, 2, 0))

        # Denormalize from [-1, 1] to [0, 255]
        image = (image * 0.5 + 0.5) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def _relative_kp(
        self,
        kp_source: Dict,
        kp_driving: Dict,
        kp_driving_initial: Dict,
        adapt_movement_scale: bool = True
    ) -> Dict:
        """
        Compute relative keypoints for better expression transfer

        Uses TPS (Thin Plate Spline) for smooth motion field
        """
        kp_new = {k: v for k, v in kp_driving.items()}

        # Relative movement
        kp_value_diff = kp_driving['value'] - kp_driving_initial['value']
        kp_new['value'] = kp_value_diff + kp_source['value']

        if adapt_movement_scale:
            # Adapt movement scale based on object size
            source_area = torch.abs(kp_source['jacobian'][:, :, 0, 0] * kp_source['jacobian'][:, :, 1, 1])
            driving_area = torch.abs(kp_driving_initial['jacobian'][:, :, 0, 0] * kp_driving_initial['jacobian'][:, :, 1, 1])
            movement_scale = torch.sqrt(source_area) / torch.sqrt(driving_area + 1e-8)
            movement_scale = movement_scale.unsqueeze(-1)
            kp_new['value'] = movement_scale * kp_value_diff + kp_source['value']

        return kp_new

    def extract_keypoints(self, image: np.ndarray) -> Dict:
        """
        Extract FOMM keypoints from an image

        Args:
            image: RGB image (H, W, 3), values 0-255

        Returns:
            Dictionary with keypoints and jacobians
        """
        image_tensor = self._preprocess_image(image).to(self.device)

        with torch.no_grad():
            kp = self.kp_detector(image_tensor)

        # Convert to CPU numpy for storage
        kp_numpy = {
            'value': kp['value'].cpu().numpy(),
            'jacobian': kp['jacobian'].cpu().numpy() if 'jacobian' in kp else None
        }

        return kp_numpy

    def visualize_keypoints(
        self,
        image: np.ndarray,
        keypoints: Dict = None
    ) -> np.ndarray:
        """
        Visualize keypoints on image

        Args:
            image: RGB image
            keypoints: Keypoint dict (if None, will extract)

        Returns:
            Image with keypoints drawn
        """
        if keypoints is None:
            keypoints = self.extract_keypoints(image)

        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # Get keypoint locations (normalized -1 to 1)
        kp_array = keypoints['value']
        if isinstance(kp_array, torch.Tensor):
            kp_array = kp_array.cpu().numpy()

        # Remove batch dimension if present
        if kp_array.ndim == 3:
            kp_array = kp_array[0]

        # Convert to pixel coordinates
        kp_pixels = (kp_array + 1) / 2  # [-1, 1] -> [0, 1]
        kp_pixels[:, 0] *= w
        kp_pixels[:, 1] *= h

        # Draw keypoints
        for i, (x, y) in enumerate(kp_pixels):
            cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(
                vis_image,
                str(i),
                (int(x) + 5, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1
            )

        return vis_image


class FaceExpressionExtractor:
    """Extract facial expressions from reference video for FOMM"""

    def __init__(self, fomm_service: FOMMService):
        self.fomm = fomm_service

    def extract_from_video(
        self,
        video_path: str,
        dancer_track: List[Dict]
    ) -> List[Dict]:
        """
        Extract facial expression keypoints from video

        Args:
            video_path: Path to video file
            dancer_track: List of {frame, bbox} for one dancer

        Returns:
            List of {frame, keypoints, bbox} dicts
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        expression_sequence = []

        for track_entry in dancer_track:
            frame_idx = track_entry['frame']
            bbox = track_entry['bbox']

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Crop to dancer (assuming face is in upper portion)
            x1, y1, x2, y2 = map(int, bbox)
            dancer_crop = frame[y1:y2, x1:x2]

            # Detect face in crop (use MediaPipe or dlib)
            face_bbox = self._detect_face(dancer_crop)

            if face_bbox is None:
                continue

            # Crop to face
            fx1, fy1, fx2, fy2 = face_bbox
            face_crop = dancer_crop[fy1:fy2, fx1:fx2]

            # Extract FOMM keypoints
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            keypoints = self.fomm.extract_keypoints(face_crop_rgb)

            expression_sequence.append({
                'frame': frame_idx,
                'keypoints': keypoints,
                'face_bbox': face_bbox,
                'body_bbox': bbox
            })

        cap.release()
        return expression_sequence

    def _detect_face(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect face bounding box in image

        Returns:
            (x1, y1, x2, y2) or None
        """
        try:
            import mediapipe as mp

            mp_face_detection = mp.solutions.face_detection

            with mp_face_detection.FaceDetection(
                model_selection=1,  # 0=short-range, 1=full-range
                min_detection_confidence=0.5
            ) as face_detection:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_image)

                if not results.detections:
                    return None

                # Get first face
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                h, w = image.shape[:2]
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                return (x1, y1, x2, y2)

        except ImportError:
            print("MediaPipe not available, using fallback face detection")
            # Fallback: assume face is in upper 40% of image
            h, w = image.shape[:2]
            return (0, 0, w, int(h * 0.4))
