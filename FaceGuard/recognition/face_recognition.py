"""Face detection and identification module for FaceGuard.

Implements high-speed face detection using OpenCV YuNet ONNX,
and deep facial feature recognition using OpenCV SFace ONNX.
Distinguishes between:
  - 'BHAVYA' (Authorized)
  - 'UNKNOWN' (Unauthorized)
  - 'NO_FACE' (Nobody detected)
  - 'MULTIPLE_FACES' (More than one person in frame)
"""

from dataclasses import dataclass
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    BHAVYA_ENROLLMENT_FILE,
    ENROLLMENT_DIR,
    FACE_DETECTION_MODEL_PATH,
    FACE_NMS_THRESHOLD,
    FACE_RECOGNITION_MODEL_PATH,
    FACE_SCORE_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    RECOGNITION_COSINE_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Represents an individual face identified by the face detector.

    Attributes:
        bbox: Bounding box tuple (x, y, width, height) in pixel coordinates.
        confidence: Detection confidence score between 0.0 and 1.0.
        landmarks: Coordinates of 5 facial landmarks:
                   [right_eye, left_eye, nose_tip, right_mouth_corner, left_mouth_corner].
        raw: Complete 15-element raw detection vector needed for alignment.
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: List[Tuple[int, int]]
    raw: Optional[np.ndarray] = None


@dataclass
class FaceDetectionResult:
    """Aggregates all face detections within a single video frame.

    Attributes:
        face_count: Total number of detected faces.
        faces: List of DetectedFace objects.
    """

    face_count: int
    faces: List[DetectedFace]

    @property
    def has_face(self) -> bool:
        """Return True if at least one face was detected."""
        return self.face_count > 0

    @property
    def is_single_face(self) -> bool:
        """Return True if exactly one face was detected."""
        return self.face_count == 1

    @property
    def is_multiple_faces(self) -> bool:
        """Return True if more than one face was detected."""
        return self.face_count > 1


class FaceDetector:
    """Detects faces in images using OpenCV's native YuNet deep-learning detector."""

    def __init__(
        self,
        model_path: str = FACE_DETECTION_MODEL_PATH,
        score_threshold: float = FACE_SCORE_THRESHOLD,
        nms_threshold: float = FACE_NMS_THRESHOLD,
        input_size: Tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT),
    ) -> None:
        """Initialize the YuNet face detector.

        Args:
            model_path: Path to the YuNet ONNX model file.
            score_threshold: Confidence threshold for filtering weak detections.
            nms_threshold: Non-maximum suppression threshold to eliminate duplicate boxes.
            input_size: Initial (width, height) expected by the detector.
        """
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._current_input_size = input_size

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Face detection model not found at '{self.model_path}'. "
                "Ensure the YuNet ONNX model is placed in data/models/."
            )

        logger.info("Initializing YuNet face detector from '%s'...", self.model_path)
        self._detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self._current_input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

    def detect(self, frame: Optional[np.ndarray]) -> FaceDetectionResult:
        """Detect faces in a given image frame.

        Args:
            frame: BGR image frame as a NumPy array (or None).

        Returns:
            FaceDetectionResult containing count, status flags, and detected face details.
        """
        if frame is None:
            return FaceDetectionResult(face_count=0, faces=[])

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Received invalid or empty image frame.")
            return FaceDetectionResult(face_count=0, faces=[])

        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return FaceDetectionResult(face_count=0, faces=[])

        # Ensure frame input size matches detector configuration dynamically
        if (frame_width, frame_height) != self._current_input_size:
            self._current_input_size = (frame_width, frame_height)
            self._detector.setInputSize(self._current_input_size)

        try:
            _, raw_detections = self._detector.detect(frame)
        except Exception as err:
            logger.error("Error executing YuNet face detection: %s", err)
            return FaceDetectionResult(face_count=0, faces=[])

        if raw_detections is None or len(raw_detections) == 0:
            return FaceDetectionResult(face_count=0, faces=[])

        detected_faces: List[DetectedFace] = []
        for face_data in raw_detections:
            raw_x, raw_y, raw_w, raw_h = face_data[0:4]
            confidence = float(face_data[14])

            # Clip bounding boxes to frame boundaries to prevent negative/out-of-bounds indices
            x = max(0, int(raw_x))
            y = max(0, int(raw_y))
            w = max(1, min(int(raw_w), frame_width - x))
            h = max(1, min(int(raw_h), frame_height - y))

            landmarks: List[Tuple[int, int]] = []
            for i in range(4, 14, 2):
                lx = int(face_data[i])
                ly = int(face_data[i + 1])
                landmarks.append((lx, ly))

            detected_faces.append(
                DetectedFace(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    landmarks=landmarks,
                    raw=face_data,
                )
            )

        return FaceDetectionResult(
            face_count=len(detected_faces),
            faces=detected_faces,
        )

    def draw_detections(
        self,
        frame: np.ndarray,
        result: FaceDetectionResult,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and confidence annotations on a copy of the frame."""
        annotated = frame.copy()
        for face in result.faces:
            x, y, w, h = face.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, thickness)
            label = f"Face: {face.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1,
            )
            for lx, ly in face.landmarks:
                cv2.circle(annotated, (lx, ly), 2, (0, 0, 255), -1)

        return annotated


class FaceRecognizer:
    """Handles face recognition, feature extraction, and Bhavya vs Unknown identification.

    Uses OpenCV's native SFace model for 128-dimensional facial embedding extraction
    and cosine similarity matching.
    """

    def __init__(
        self,
        model_path: str = FACE_RECOGNITION_MODEL_PATH,
        known_faces_file: str = BHAVYA_ENROLLMENT_FILE,
        threshold: float = RECOGNITION_COSINE_THRESHOLD,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        """Initialize face recognizer.

        Args:
            model_path: Path to SFace ONNX recognition model.
            known_faces_file: Path to serialized reference embeddings (.npy).
            threshold: Cosine similarity threshold for authorized match.
            detector: FaceDetector instance (instantiated if None).
        """
        self.model_path = model_path
        self.known_faces_file = known_faces_file
        self.threshold = threshold
        self.detector = detector if detector is not None else FaceDetector()

        self.known_embeddings: Optional[np.ndarray] = None
        self._recognizer: Optional[object] = None

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Face recognition model not found at '{self.model_path}'. "
                "Ensure face_recognition_sface_2021dec.onnx is in data/models/."
            )

        logger.info("Initializing SFace face recognizer from '%s'...", self.model_path)
        self._recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config="",
        )

        # Attempt to load enrolled faces on startup
        self.load_known_faces()

    @property
    def is_enrolled(self) -> bool:
        """Return True if authorized reference embeddings are loaded."""
        return self.known_embeddings is not None and len(self.known_embeddings) > 0

    def load_known_faces(self) -> bool:
        """Load enrolled face embeddings from disk.

        Returns:
            bool: True if loaded successfully, False if file doesn't exist or is invalid.
        """
        if not os.path.exists(self.known_faces_file):
            logger.info("No enrolled face file found at '%s'.", self.known_faces_file)
            self.known_embeddings = None
            return False

        try:
            embeddings = np.load(self.known_faces_file)
            if embeddings.size == 0:
                logger.warning("Enrolled embeddings file '%s' is empty.", self.known_faces_file)
                self.known_embeddings = None
                return False

            # Ensure 2D shape (N, 128)
            if embeddings.ndim == 1:
                embeddings = np.expand_dims(embeddings, axis=0)

            self.known_embeddings = embeddings
            logger.info(
                "Successfully loaded %d reference embedding(s) for Bhavya.",
                len(self.known_embeddings),
            )
            return True
        except Exception as err:
            logger.error("Failed to load enrolled face embeddings from '%s': %s", self.known_faces_file, err)
            self.known_embeddings = None
            return False

    def enroll_face(self, embeddings: List[np.ndarray]) -> bool:
        """Save a list of facial feature vectors as Bhavya's authorized reference representation.

        Args:
            embeddings: List of 128-dimensional embedding vectors.

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        if not embeddings:
            logger.warning("Cannot enroll: No embeddings provided.")
            return False

        try:
            os.makedirs(os.path.dirname(self.known_faces_file), exist_ok=True)
            # Stack all sample embeddings (N, 128)
            stacked = np.vstack(embeddings).astype(np.float32)
            np.save(self.known_faces_file, stacked)
            self.known_embeddings = stacked
            logger.info(
                "Successfully enrolled face with %d reference sample(s) to '%s'.",
                len(stacked),
                self.known_faces_file,
            )
            return True
        except Exception as err:
            logger.error("Failed to save enrollment file '%s': %s", self.known_faces_file, err)
            return False

    def extract_features(self, frame: np.ndarray, face: DetectedFace) -> Optional[np.ndarray]:
        """Align face crop and extract 128-D normalized feature vector.

        Args:
            frame: Full BGR video frame.
            face: DetectedFace containing raw detection parameters.

        Returns:
            Optional[np.ndarray]: 128-D feature vector, or None if extraction fails.
        """
        if self._recognizer is None or face.raw is None:
            return None

        try:
            aligned = self._recognizer.alignCrop(frame, face.raw)
            feat = self._recognizer.feature(aligned)
            return feat
        except Exception as err:
            logger.error("Error extracting facial features: %s", err)
            return None

    def identify_face(self, frame: Optional[np.ndarray]) -> Tuple[str, float]:
        """Analyze frame, detect faces, and classify identity.

        Returns:
            Tuple[str, float]: (identity, confidence_or_similarity)
              - ('NO_FACE', 0.0) if no face detected or frame is None
              - ('BHAVYA', similarity) if matched with Bhavya above threshold
              - ('UNKNOWN', similarity) if face does not match Bhavya
              - ('MULTIPLE_FACES', count) if more than one face is in frame
        """
        if frame is None:
            return ("NO_FACE", 0.0)

        detection_result = self.detector.detect(frame)

        if not detection_result.has_face:
            return ("NO_FACE", 0.0)

        if detection_result.is_multiple_faces:
            # When multiple faces appear, flag MULTIPLE_FACES for security evaluation
            return ("MULTIPLE_FACES", float(detection_result.face_count))

        # Exactly 1 face
        target_face = detection_result.faces[0]
        feat = self.extract_features(frame, target_face)

        if feat is None:
            return ("UNKNOWN", 0.0)

        if not self.is_enrolled or self.known_embeddings is None:
            logger.debug("No reference face enrolled. Classifying face as UNKNOWN.")
            return ("UNKNOWN", 0.0)

        # Match against all enrolled sample vectors and pick the best cosine similarity
        best_similarity = -1.0
        for ref_feat in self.known_embeddings:
            ref_feat_2d = np.expand_dims(ref_feat, axis=0) if ref_feat.ndim == 1 else ref_feat
            sim = float(
                self._recognizer.match(
                    ref_feat_2d,
                    feat,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
            if sim > best_similarity:
                best_similarity = sim

        if best_similarity >= self.threshold:
            return ("BHAVYA", best_similarity)
        else:
            return ("UNKNOWN", best_similarity)
