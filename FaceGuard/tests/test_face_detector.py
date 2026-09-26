"""Unit tests for FaceDetector module using unittest and mocks."""

import unittest
from unittest.mock import MagicMock
import numpy as np

from recognition.face_recognition import (
    DetectedFace,
    FaceDetectionResult,
    FaceDetector,
)
from config import FACE_DETECTION_MODEL_PATH


class TestFaceDetector(unittest.TestCase):
    """Test suite for FaceDetector component."""

    def setUp(self) -> None:
        """Initialize FaceDetector instance."""
        self.detector = FaceDetector(model_path=FACE_DETECTION_MODEL_PATH)

    def test_initialization_with_missing_model_raises_error(self) -> None:
        """Verify FileNotFoundError is raised if model file does not exist."""
        with self.assertRaises(FileNotFoundError):
            FaceDetector(model_path="data/models/non_existent_model.onnx")

    def test_detect_none_frame(self) -> None:
        """Verify detector handles None frame without crashing."""
        result = self.detector.detect(None)
        self.assertIsInstance(result, FaceDetectionResult)
        self.assertEqual(result.face_count, 0)
        self.assertFalse(result.has_face)
        self.assertFalse(result.is_single_face)
        self.assertFalse(result.is_multiple_faces)
        self.assertEqual(len(result.faces), 0)

    def test_detect_invalid_frame_types(self) -> None:
        """Verify detector handles non-ndarray or zero-size frames safely."""
        for invalid in ["not_an_image", 12345, np.zeros((0, 0, 3), dtype=np.uint8)]:
            with self.subTest(invalid=type(invalid)):
                result = self.detector.detect(invalid)  # type: ignore[arg-type]
                self.assertEqual(result.face_count, 0)
                self.assertFalse(result.has_face)

    def test_detect_blank_frame_no_faces(self) -> None:
        """Verify detector finds 0 faces on blank black frame."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(blank_frame)

        self.assertEqual(result.face_count, 0)
        self.assertFalse(result.has_face)
        self.assertFalse(result.is_single_face)
        self.assertFalse(result.is_multiple_faces)
        self.assertEqual(result.faces, [])

    def test_detect_single_face_mocked(self) -> None:
        """Verify single face detection parsing and properties."""
        fake_face = np.array([
            [100.0, 80.0, 150.0, 180.0,
             130.0, 120.0, 200.0, 120.0, 165.0, 160.0, 140.0, 210.0, 190.0, 210.0,
             0.95]
        ], dtype=np.float32)

        mock_backend = MagicMock()
        mock_backend.detect.return_value = (1, fake_face)
        self.detector._detector = mock_backend

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)

        self.assertEqual(result.face_count, 1)
        self.assertTrue(result.has_face)
        self.assertTrue(result.is_single_face)
        self.assertFalse(result.is_multiple_faces)

        face = result.faces[0]
        self.assertIsInstance(face, DetectedFace)
        self.assertEqual(face.bbox, (100, 80, 150, 180))
        self.assertAlmostEqual(face.confidence, 0.95, places=2)
        self.assertEqual(len(face.landmarks), 5)
        self.assertEqual(face.landmarks[0], (130, 120))  # right eye
        self.assertEqual(face.landmarks[2], (165, 160))  # nose tip

    def test_detect_multiple_faces_mocked(self) -> None:
        """Verify multiple face detection parsing and count."""
        fake_faces = np.array([
            [50.0, 50.0, 100.0, 100.0, 70.0, 70.0, 110.0, 70.0, 90.0, 90.0, 80.0, 120.0, 100.0, 120.0, 0.92],
            [300.0, 100.0, 120.0, 120.0, 320.0, 120.0, 380.0, 120.0, 350.0, 150.0, 330.0, 180.0, 370.0, 180.0, 0.88],
        ], dtype=np.float32)

        mock_backend = MagicMock()
        mock_backend.detect.return_value = (1, fake_faces)
        self.detector._detector = mock_backend

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)

        self.assertEqual(result.face_count, 2)
        self.assertTrue(result.has_face)
        self.assertFalse(result.is_single_face)
        self.assertTrue(result.is_multiple_faces)
        self.assertEqual(len(result.faces), 2)

    def test_dynamic_frame_resizing(self) -> None:
        """Verify detector updates internal input size when frame dimensions change."""
        frame_hd = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_backend = MagicMock()
        mock_backend.detect.return_value = (1, None)
        self.detector._detector = mock_backend

        self.detector.detect(frame_hd)
        mock_backend.setInputSize.assert_called_with((1280, 720))

    def test_bounding_box_clipping(self) -> None:
        """Verify bounding boxes extending outside frame boundaries are safely clipped."""
        fake_face = np.array([
            [-20.0, -10.0, 700.0, 550.0,
             100.0, 100.0, 200.0, 100.0, 150.0, 150.0, 120.0, 200.0, 180.0, 200.0,
             0.85]
        ], dtype=np.float32)

        mock_backend = MagicMock()
        mock_backend.detect.return_value = (1, fake_face)
        self.detector._detector = mock_backend

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)

        face = result.faces[0]
        x, y, w, h = face.bbox
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 640)
        self.assertLessEqual(y + h, 480)

    def test_draw_detections_returns_valid_image(self) -> None:
        """Verify draw_detections draws on image and returns ndarray copy."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detected_face = DetectedFace(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            landmarks=[(20, 20), (40, 20), (30, 30), (25, 45), (35, 45)],
        )
        result = FaceDetectionResult(face_count=1, faces=[detected_face])

        annotated = self.detector.draw_detections(frame, result)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
