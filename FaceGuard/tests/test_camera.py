"""Unit tests for CameraStream module using unittest and mocks."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from camera.camera import CameraStream
from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH


class TestCameraStream(unittest.TestCase):
    """Test suite for CameraStream webcam acquisition component."""

    def setUp(self) -> None:
        """Create a fresh CameraStream instance before each test."""
        self.camera = CameraStream(
            camera_index=CAMERA_INDEX,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
        )

    def tearDown(self) -> None:
        """Ensure camera resource is stopped after each test."""
        self.camera.stop()

    def test_initial_state(self) -> None:
        """Verify initial camera properties."""
        self.assertEqual(self.camera.camera_index, CAMERA_INDEX)
        self.assertEqual(self.camera.width, FRAME_WIDTH)
        self.assertEqual(self.camera.height, FRAME_HEIGHT)
        self.assertFalse(self.camera.is_running)
        self.assertFalse(self.camera.is_opened)
        self.assertIsNone(self.camera.cap)

    @patch("cv2.VideoCapture")
    def test_start_success(self, mock_videocapture: MagicMock) -> None:
        """Verify successful webcam initialization sets properties and flags."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap

        success = self.camera.start()

        self.assertTrue(success)
        self.assertTrue(self.camera.is_running)
        self.assertTrue(self.camera.is_opened)
        mock_cap.set.assert_any_call(3, FRAME_WIDTH)   # CAP_PROP_FRAME_WIDTH = 3
        mock_cap.set.assert_any_call(4, FRAME_HEIGHT)  # CAP_PROP_FRAME_HEIGHT = 4

    @patch("cv2.VideoCapture")
    def test_start_failure_camera_unavailable(self, mock_videocapture: MagicMock) -> None:
        """Verify graceful failure when webcam cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        success = self.camera.start()

        self.assertFalse(success)
        self.assertFalse(self.camera.is_running)
        self.assertIsNone(self.camera.cap)

    @patch("cv2.VideoCapture")
    def test_read_frame_success(self, mock_videocapture: MagicMock) -> None:
        """Verify valid frame read returns NumPy ndarray."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_videocapture.return_value = mock_cap

        self.camera.start()
        frame = self.camera.read_frame()

        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3))

    def test_read_frame_when_stopped_returns_none(self) -> None:
        """Verify read_frame returns None when stream has not been started."""
        frame = self.camera.read_frame()
        self.assertIsNone(frame)

    @patch("cv2.VideoCapture")
    def test_read_frame_failure_returns_none(self, mock_videocapture: MagicMock) -> None:
        """Verify read_frame handles read failure gracefully."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_videocapture.return_value = mock_cap

        self.camera.start()
        frame = self.camera.read_frame()

        self.assertIsNone(frame)

    @patch("cv2.VideoCapture")
    def test_stop_releases_resource(self, mock_videocapture: MagicMock) -> None:
        """Verify stop cleanly releases capture device."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap

        self.camera.start()
        self.camera.stop()

        mock_cap.release.assert_called_once()
        self.assertFalse(self.camera.is_running)
        self.assertIsNone(self.camera.cap)

    @patch("cv2.VideoCapture")
    def test_context_manager(self, mock_videocapture: MagicMock) -> None:
        """Verify context manager usage starts and stops camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap

        with CameraStream(camera_index=0) as cam:
            self.assertTrue(cam.is_running)

        mock_cap.release.assert_called_once()
        self.assertFalse(cam.is_running)


if __name__ == "__main__":
    unittest.main()
