"""Unit tests for FaceRecognizer module using unittest and mocks."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np

from recognition.face_recognition import (
    DetectedFace,
    FaceDetectionResult,
    FaceRecognizer,
)
from config import FACE_RECOGNITION_MODEL_PATH


class TestFaceRecognizer(unittest.TestCase):
    """Test suite for FaceRecognizer component."""

    def setUp(self) -> None:
        """Create temporary enrollment file and recognizer instance."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.enroll_file = os.path.join(self.temp_dir.name, "bhavya.npy")

        # Mock detector to isolate recognizer tests
        self.mock_detector = MagicMock()
        self.recognizer = FaceRecognizer(
            model_path=FACE_RECOGNITION_MODEL_PATH,
            known_faces_file=self.enroll_file,
            threshold=0.38,
            detector=self.mock_detector,
        )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_missing_model_raises_error(self) -> None:
        """Verify FileNotFoundError if SFace ONNX model is missing."""
        with self.assertRaises(FileNotFoundError):
            FaceRecognizer(model_path="data/models/missing_model.onnx")

    def test_initially_not_enrolled(self) -> None:
        """Verify recognizer is not enrolled when file doesn't exist."""
        self.assertFalse(self.recognizer.is_enrolled)
        self.assertIsNone(self.recognizer.known_embeddings)

    def test_enroll_face_saves_and_loads(self) -> None:
        """Verify enroll_face saves embeddings and loads them successfully."""
        sample1 = np.ones((1, 128), dtype=np.float32)
        sample2 = np.ones((1, 128), dtype=np.float32) * 0.9

        success = self.recognizer.enroll_face([sample1, sample2])
        self.assertTrue(success)
        self.assertTrue(self.recognizer.is_enrolled)
        self.assertEqual(len(self.recognizer.known_embeddings), 2)
        self.assertTrue(os.path.exists(self.enroll_file))

        # Test reloading from disk
        rec2 = FaceRecognizer(
            model_path=FACE_RECOGNITION_MODEL_PATH,
            known_faces_file=self.enroll_file,
            threshold=0.38,
            detector=self.mock_detector,
        )
        self.assertTrue(rec2.is_enrolled)
        self.assertEqual(len(rec2.known_embeddings), 2)

    def test_identify_none_frame(self) -> None:
        """Verify None frame returns ('NO_FACE', 0.0)."""
        identity, score = self.recognizer.identify_face(None)
        self.assertEqual(identity, "NO_FACE")
        self.assertEqual(score, 0.0)

    def test_identify_no_faces_detected(self) -> None:
        """Verify frame with 0 detected faces returns ('NO_FACE', 0.0)."""
        self.mock_detector.detect.return_value = FaceDetectionResult(face_count=0, faces=[])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        identity, score = self.recognizer.identify_face(frame)
        self.assertEqual(identity, "NO_FACE")
        self.assertEqual(score, 0.0)

    def test_identify_multiple_faces(self) -> None:
        """Verify frame with >1 faces returns ('MULTIPLE_FACES', count)."""
        self.mock_detector.detect.return_value = FaceDetectionResult(
            face_count=2,
            faces=[
                DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9, landmarks=[]),
                DetectedFace(bbox=(100, 10, 50, 50), confidence=0.85, landmarks=[]),
            ],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        identity, score = self.recognizer.identify_face(frame)
        self.assertEqual(identity, "MULTIPLE_FACES")
        self.assertEqual(score, 2.0)

    def test_identify_authorized_bhavya_match(self) -> None:
        """Verify face matching enrolled embedding is identified as BHAVYA."""
        # Enrol synthetic normalized vector
        vec = np.random.randn(1, 128).astype(np.float32)
        vec /= np.linalg.norm(vec)
        self.recognizer.enroll_face([vec])

        # Mock detector returning 1 face
        raw_face = np.zeros(15, dtype=np.float32)
        face = DetectedFace(bbox=(50, 50, 100, 100), confidence=0.95, landmarks=[], raw=raw_face)
        self.mock_detector.detect.return_value = FaceDetectionResult(face_count=1, faces=[face])

        # Mock feature extraction to return the exact enrolled vector
        mock_rec_backend = MagicMock()
        mock_rec_backend.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_rec_backend.feature.return_value = vec
        mock_rec_backend.match.return_value = 0.95  # > threshold 0.38
        self.recognizer._recognizer = mock_rec_backend

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        identity, similarity = self.recognizer.identify_face(frame)

        self.assertEqual(identity, "BHAVYA")
        self.assertAlmostEqual(similarity, 0.95, places=2)

    def test_identify_unknown_person(self) -> None:
        """Verify face not matching enrolled embedding is classified as UNKNOWN."""
        # Enrol reference vector
        vec = np.random.randn(1, 128).astype(np.float32)
        self.recognizer.enroll_face([vec])

        # Mock detector returning 1 face
        raw_face = np.zeros(15, dtype=np.float32)
        face = DetectedFace(bbox=(50, 50, 100, 100), confidence=0.95, landmarks=[], raw=raw_face)
        self.mock_detector.detect.return_value = FaceDetectionResult(face_count=1, faces=[face])

        # Mock matching with low similarity (0.15 < 0.38)
        mock_rec_backend = MagicMock()
        mock_rec_backend.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_rec_backend.feature.return_value = np.random.randn(1, 128).astype(np.float32)
        mock_rec_backend.match.return_value = 0.15
        self.recognizer._recognizer = mock_rec_backend

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        identity, similarity = self.recognizer.identify_face(frame)

        self.assertEqual(identity, "UNKNOWN")
        self.assertAlmostEqual(similarity, 0.15, places=2)

    def test_identify_when_not_enrolled_returns_unknown(self) -> None:
        """Verify any detected face without enrolled reference returns UNKNOWN."""
        raw_face = np.zeros(15, dtype=np.float32)
        face = DetectedFace(bbox=(50, 50, 100, 100), confidence=0.95, landmarks=[], raw=raw_face)
        self.mock_detector.detect.return_value = FaceDetectionResult(face_count=1, faces=[face])

        mock_rec_backend = MagicMock()
        mock_rec_backend.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_rec_backend.feature.return_value = np.random.randn(1, 128).astype(np.float32)
        self.recognizer._recognizer = mock_rec_backend

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        identity, similarity = self.recognizer.identify_face(frame)

        self.assertEqual(identity, "UNKNOWN")
        self.assertEqual(similarity, 0.0)


if __name__ == "__main__":
    unittest.main()
