"""
Face detection and alignment for deepfake detection preprocessing.
Uses RetinaFace for robust multi-scale face detection.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detect and align faces in images using RetinaFace.
    Produces aligned face crops for consistent model input.
    """

    def __init__(self, model_name: str = "retinaface", device: str = "cuda"):
        """
        Initialize face detector.

        Args:
            model_name: Detection model ("retinaface" or "mtcnn")
            device: "cuda" or "cpu"
        """
        self.model_name = model_name
        self.device = device

        if model_name == "retinaface":
            try:
                from retinaface import RetinaFace
                self.detector = RetinaFace.RetinaFace(gpu=device == "cuda")
            except ImportError:
                logger.warning("RetinaFace not available, falling back to OpenCV cascade")
                self.detector = None
        else:
            logger.warning(f"Model {model_name} not supported, using OpenCV cascade")
            self.detector = None

        # Fallback: OpenCV cascade classifier
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of face detections with bounding boxes and landmarks
        """
        if self.detector is not None:
            try:
                faces = self.detector.detect_faces(image)
                return [self._parse_retinaface_detection(f) for f in faces]
            except Exception as e:
                logger.debug(f"RetinaFace detection failed: {e}")

        # Fallback to OpenCV cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 4)

        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.5,
                'landmarks': None
            })
        return detections

    def _parse_retinaface_detection(self, detection: dict) -> dict:
        """Parse RetinaFace detection output."""
        face_info = detection["facial_area"]
        landmarks = detection.get("landmarks", {})

        # Convert landmarks dict to array if available
        landmark_array = None
        if landmarks:
            try:
                landmark_array = np.array([
                    landmarks.get(i, [0, 0]) for i in range(5)
                ])
            except:
                landmark_array = None

        return {
            'bbox': [face_info[0], face_info[1], face_info[2], face_info[3]],
            'confidence': detection.get("confidence", 0.9),
            'landmarks': landmark_array
        }

    def align_face(
        self,
        image: np.ndarray,
        face_bbox: List[int],
        landmarks: Optional[np.ndarray] = None,
        output_size: int = 224,
        expand_ratio: float = 0.0
    ) -> Tuple[np.ndarray, dict]:
        """
        Extract and align face crop from image.

        Args:
            image: Input image
            face_bbox: Bounding box [x1, y1, x2, y2]
            landmarks: Face landmarks for precise alignment
            output_size: Size of output crop
            expand_ratio: Expand bbox by this ratio (0.0 = no expansion)

        Returns:
            Tuple of (aligned_face, alignment_info)
        """
        x1, y1, x2, y2 = face_bbox
        h, w = image.shape[:2]

        # Expand bounding box if requested
        if expand_ratio > 0:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, int(cx - bw * (1 + expand_ratio) / 2))
            x2 = min(w, int(cx + bw * (1 + expand_ratio) / 2))
            y1 = max(0, int(cy - bh * (1 + expand_ratio) / 2))
            y2 = min(h, int(cy + bh * (1 + expand_ratio) / 2))

        # Ensure valid bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        # Extract face region
        face = image[y1:y2, x1:x2]

        # Align using landmarks if available
        if landmarks is not None and len(landmarks) >= 2:
            aligned_face = self._align_with_landmarks(
                face, landmarks, output_size
            )
        else:
            # Simple resize without alignment
            aligned_face = cv2.resize(face, (output_size, output_size))

        alignment_info = {
            'original_bbox': face_bbox,
            'expanded_bbox': [x1, y1, x2, y2],
            'output_size': output_size,
            'used_landmarks': landmarks is not None
        }

        return aligned_face, alignment_info

    def _align_with_landmarks(
        self,
        face: np.ndarray,
        landmarks: np.ndarray,
        output_size: int
    ) -> np.ndarray:
        """Align face using facial landmarks."""
        # Use eye landmarks for alignment (typically indices 0-1)
        if len(landmarks) >= 2:
            left_eye = landmarks[0]
            right_eye = landmarks[1]

            # Calculate angle between eyes
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.arctan2(dy, dx) * 180 / np.pi

            # Get rotation matrix
            h, w = face.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation and resize
            rotated = cv2.warpAffine(face, M, (w, h))
            aligned = cv2.resize(rotated, (output_size, output_size))
            return aligned

        return cv2.resize(face, (output_size, output_size))


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline: video → frames → face detection → aligned crops.
    """

    def __init__(
        self,
        detector_model: str = "retinaface",
        face_size: int = 224,
        device: str = "cuda"
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            detector_model: Face detector model type
            face_size: Output face crop size
            device: Computation device
        """
        self.detector = FaceDetector(detector_model, device)
        self.face_size = face_size
        self.device = device

    def process_frame(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.5,
        expand_ratio: float = 0.1
    ) -> Tuple[List[np.ndarray], List[dict]]:
        """
        Process single frame: detect faces, align, and extract crops.

        Args:
            frame: Input frame (BGR format)
            min_confidence: Minimum detection confidence
            expand_ratio: Expand face bbox by this ratio

        Returns:
            Tuple of (face_crops, face_info)
        """
        faces = self.detector.detect_faces(frame)

        face_crops = []
        face_info = []

        for detection in faces:
            if detection['confidence'] < min_confidence:
                continue

            aligned_face, info = self.detector.align_face(
                frame,
                detection['bbox'],
                detection.get('landmarks'),
                self.face_size,
                expand_ratio
            )

            face_crops.append(aligned_face)
            info['confidence'] = detection['confidence']
            face_info.append(info)

        return face_crops, face_info

    def extract_frames_from_video(
        self,
        video_path: str,
        num_frames: int = 8,
        sampling_strategy: str = "uniform"
    ) -> List[np.ndarray]:
        """
        Extract specific number of frames from video.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            sampling_strategy: "uniform", "keyframe", or "motion"

        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            logger.warning(f"Could not read video: {video_path}")
            return []

        frames = []

        if sampling_strategy == "uniform":
            # Sample uniformly across video
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif sampling_strategy == "keyframe":
            # Extract first and every nth frame
            frame_indices = list(range(0, total_frames, max(1, total_frames // num_frames)))[:num_frames]
        else:
            # Random sampling (for data augmentation)
            frame_indices = sorted(np.random.choice(total_frames, num_frames, replace=False))

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frames.append(frame)
            else:
                logger.debug(f"Failed to read frame {idx} from {video_path}")

        cap.release()
        return frames

    def process_video(
        self,
        video_path: str,
        num_frames: int = 8,
        sampling_strategy: str = "uniform",
        min_confidence: float = 0.5,
        expand_ratio: float = 0.1
    ) -> Tuple[List[np.ndarray], List[dict]]:
        """
        Complete pipeline: extract frames, detect faces, align, return crops.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            sampling_strategy: Frame sampling strategy
            min_confidence: Minimum face detection confidence
            expand_ratio: Expand face bbox ratio

        Returns:
            Tuple of (all_face_crops, processing_info)
        """
        frames = self.extract_frames_from_video(
            video_path, num_frames, sampling_strategy
        )

        all_faces = []
        processing_info = []

        for frame_idx, frame in enumerate(frames):
            faces, info = self.process_frame(
                frame, min_confidence, expand_ratio
            )

            all_faces.extend(faces)
            for face_info in info:
                face_info['frame_index'] = frame_idx
                processing_info.append(face_info)

        return all_faces, processing_info
