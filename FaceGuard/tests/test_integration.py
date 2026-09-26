"""Integration test simulating continuous person transitions and ESP LED command transmission."""

import unittest
from unittest.mock import MagicMock
import numpy as np

from esp.esp_controller import ESPController
from recognition.face_recognition import FaceRecognizer
from security.security_manager import SecurityManager
from config import STATE_BHAVYA, STATE_UNKNOWN, STATE_IDLE


class TestPipelineIntegration(unittest.TestCase):
    """Integration test suite for the complete FaceGuard pipeline."""

    def test_continuous_pipeline_with_mocked_esp(self) -> None:
        """Simulate continuous operation:
        Bhavya sits -> Green ('B')
        Bhavya leaves -> OFF ('O')
        Unknown person sits -> Red ('U')
        Unknown person leaves -> OFF ('O')
        Bhavya returns -> Green ('B')
        """
        # Mock ESP serial connection
        mock_serial = MagicMock()
        mock_serial.is_open = True
        esp = ESPController(port="COM5")
        esp.connection = mock_serial

        security = SecurityManager(frames_bhavya=3, frames_unknown=4, frames_no_face=3)
        mock_recognizer = MagicMock()

        def process_frame(identity: str) -> str:
            mock_recognizer.identify_face.return_value = (identity, 0.9)
            ident, _ = mock_recognizer.identify_face(None)
            confirmed = security.evaluate_state(ident)
            esp.send_state(confirmed)
            return confirmed

        # 1. Bhavya sits down (3 frames needed)
        for _ in range(2):
            self.assertEqual(process_frame("BHAVYA"), STATE_IDLE)
        self.assertEqual(process_frame("BHAVYA"), STATE_BHAVYA)
        # Check ESP received 'B'
        mock_serial.write.assert_called_with(f"{STATE_BHAVYA}\n".encode("ascii"))
        self.assertEqual(esp.last_sent_state, STATE_BHAVYA)

        # 2. Bhavya leaves (3 empty frames needed)
        for _ in range(2):
            self.assertEqual(process_frame("NO_FACE"), STATE_BHAVYA)
        self.assertEqual(process_frame("NO_FACE"), STATE_IDLE)
        # Check ESP received 'O'
        mock_serial.write.assert_called_with(f"{STATE_IDLE}\n".encode("ascii"))
        self.assertEqual(esp.last_sent_state, STATE_IDLE)

        # 3. Unknown person sits down (4 frames needed)
        for _ in range(3):
            self.assertEqual(process_frame("UNKNOWN"), STATE_IDLE)
        self.assertEqual(process_frame("UNKNOWN"), STATE_UNKNOWN)
        # Check ESP received 'U'
        mock_serial.write.assert_called_with(f"{STATE_UNKNOWN}\n".encode("ascii"))
        self.assertEqual(esp.last_sent_state, STATE_UNKNOWN)

        # 4. Unknown person leaves (3 empty frames needed)
        for _ in range(2):
            self.assertEqual(process_frame("NO_FACE"), STATE_UNKNOWN)
        self.assertEqual(process_frame("NO_FACE"), STATE_IDLE)
        # Check ESP received 'O'
        mock_serial.write.assert_called_with(f"{STATE_IDLE}\n".encode("ascii"))
        self.assertEqual(esp.last_sent_state, STATE_IDLE)

        # 5. Bhavya returns (3 frames needed)
        for _ in range(2):
            self.assertEqual(process_frame("BHAVYA"), STATE_IDLE)
        self.assertEqual(process_frame("BHAVYA"), STATE_BHAVYA)
        # Check ESP received 'B'
        mock_serial.write.assert_called_with(f"{STATE_BHAVYA}\n".encode("ascii"))
        self.assertEqual(esp.last_sent_state, STATE_BHAVYA)


if __name__ == "__main__":
    unittest.main()
