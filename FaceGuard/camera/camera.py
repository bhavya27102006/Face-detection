"""Camera stream handler for webcam video capture.

Encapsulates webcam access, frame acquisition, and hardware resource cleanup.
Does NOT perform face detection or recognition (modular separation of concerns).
"""

import logging
from typing import Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # Handled gracefully if cv2 is not yet installed

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH

logger = logging.getLogger(__name__)


class CameraStream:
    """Manages webcam capture, frame acquisition, and cleanup."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        """Initialize camera stream parameters.

        Args:
            camera_index: Webcam device index (default: 0).
            width: Target frame width in pixels.
            height: Target frame height in pixels.
        """
        self.camera_index: int = camera_index
        self.width: int = width
        self.height: int = height
        self.cap: Optional[object] = None
        self.is_running: bool = False

    @property
    def is_opened(self) -> bool:
        """Return True if camera resource is actively opened."""
        return self.cap is not None and getattr(self.cap, "isOpened", lambda: False)()

    def start(self) -> bool:
        """Start the webcam video stream.

        Returns:
            bool: True if camera initialized and opened successfully, False otherwise.
        """
        if cv2 is None:
            logger.error("OpenCV (cv2) is not installed. Please install opencv-python.")
            return False

        if self.is_running and self.is_opened:
            logger.info("Camera stream is already active on index %d.", self.camera_index)
            return True

        logger.info("Opening webcam (index: %d, %dx%d)...", self.camera_index, self.width, self.height)

        try:
            # On Windows, DirectShow (CAP_DSHOW) initializes webcams much faster
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback to default backend if DirectShow fails
                logger.debug("CAP_DSHOW backend failed, trying default backend...")
                cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                logger.error("Unable to open camera at index %d.", self.camera_index)
                cap.release()
                self.cap = None
                self.is_running = False
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self.cap = cap
            self.is_running = True
            logger.info("Camera stream successfully started.")
            return True
        except Exception as err:
            logger.error("Unexpected error opening camera: %s", err)
            self.stop()
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Capture and return the current video frame.

        Returns:
            Optional[np.ndarray]: Captured frame as a NumPy array (BGR format),
                                  or None if capture failed or camera is inactive.
        """
        if not self.is_running or self.cap is None:
            logger.warning("Cannot read frame: Camera stream is not running.")
            return None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Failed to retrieve frame from camera device.")
                return None
            return frame
        except Exception as err:
            logger.error("Error reading frame from camera: %s", err)
            return None

    def stop(self) -> None:
        """Release the webcam hardware resource safely."""
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Camera resource released.")
            except Exception as err:
                logger.warning("Error releasing camera resource: %s", err)
            finally:
                self.cap = None

        self.is_running = False

    def __enter__(self) -> "CameraStream":
        """Context manager entrance."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit guaranteeing cleanup."""
        self.stop()
