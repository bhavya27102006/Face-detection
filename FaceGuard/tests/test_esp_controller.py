"""Unit tests for ESPController module using unittest and mocks."""

import unittest
from unittest.mock import MagicMock, patch
import serial

from esp.esp_controller import ESPController
from config import STATE_BHAVYA, STATE_UNKNOWN, STATE_IDLE, SERIAL_PORT, BAUD_RATE


class TestESPController(unittest.TestCase):
    """Test suite for ESP-12E serial communication controller."""

    def setUp(self) -> None:
        """Create a fresh ESPController instance before each test."""
        self.controller = ESPController(port=SERIAL_PORT, baud_rate=BAUD_RATE)

    def test_initial_state(self) -> None:
        """Verify controller initial properties."""
        self.assertEqual(self.controller.port, SERIAL_PORT)
        self.assertEqual(self.controller.baud_rate, BAUD_RATE)
        self.assertFalse(self.controller.is_connected)
        self.assertIsNone(self.controller.last_sent_state)

    @patch("serial.Serial")
    def test_connect_success(self, mock_serial_cls: MagicMock) -> None:
        """Verify successful connection opens port and resets buffers."""
        mock_serial_instance = MagicMock()
        mock_serial_instance.is_open = True
        mock_serial_cls.return_value = mock_serial_instance

        success = self.controller.connect()

        self.assertTrue(success)
        self.assertTrue(self.controller.is_connected)
        mock_serial_cls.assert_called_once_with(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1.0,
            write_timeout=1.0,
        )
        mock_serial_instance.reset_input_buffer.assert_called_once()
        mock_serial_instance.reset_output_buffer.assert_called_once()

    @patch("serial.Serial", side_effect=serial.SerialException("Port not found"))
    def test_connect_failure_does_not_crash(self, mock_serial_cls: MagicMock) -> None:
        """Verify connection error returns False without unhandled exception."""
        success = self.controller.connect()

        self.assertFalse(success)
        self.assertFalse(self.controller.is_connected)
        self.assertIsNone(self.controller.connection)

    @patch("serial.Serial")
    def test_send_valid_states(self, mock_serial_cls: MagicMock) -> None:
        """Verify valid states ('B', 'U', 'O') are transmitted correctly."""
        mock_serial_instance = MagicMock()
        mock_serial_instance.is_open = True
        mock_serial_cls.return_value = mock_serial_instance

        self.controller.connect()

        for state in (STATE_BHAVYA, STATE_UNKNOWN, STATE_IDLE):
            with self.subTest(state=state):
                result = self.controller.send_state(state, force=True)
                self.assertTrue(result)
                mock_serial_instance.write.assert_called_with(f"{state}\n".encode("ascii"))
                mock_serial_instance.flush.assert_called()
                self.assertEqual(self.controller.last_sent_state, state)

    @patch("serial.Serial")
    def test_send_invalid_state_rejected(self, mock_serial_cls: MagicMock) -> None:
        """Verify invalid state characters are rejected without sending."""
        mock_serial_instance = MagicMock()
        mock_serial_instance.is_open = True
        mock_serial_cls.return_value = mock_serial_instance

        self.controller.connect()

        for invalid_state in ("X", "1", "", "b", "unknown", None):
            with self.subTest(invalid=invalid_state):
                result = self.controller.send_state(invalid_state)  # type: ignore[arg-type]
                self.assertFalse(result)
                mock_serial_instance.write.assert_not_called()

    @patch("serial.Serial")
    def test_duplicate_suppression(self, mock_serial_cls: MagicMock) -> None:
        """Verify sending duplicate state is ignored unless forced."""
        mock_serial_instance = MagicMock()
        mock_serial_instance.is_open = True
        mock_serial_cls.return_value = mock_serial_instance

        self.controller.connect()

        # First send
        res1 = self.controller.send_state(STATE_BHAVYA)
        self.assertTrue(res1)
        self.assertEqual(mock_serial_instance.write.call_count, 1)

        # Duplicate send (without force)
        res2 = self.controller.send_state(STATE_BHAVYA)
        self.assertTrue(res2)
        # Should not have called write a second time
        self.assertEqual(mock_serial_instance.write.call_count, 1)

        # Forced send
        res3 = self.controller.send_state(STATE_BHAVYA, force=True)
        self.assertTrue(res3)
        self.assertEqual(mock_serial_instance.write.call_count, 2)

    def test_send_when_disconnected_fails_gracefully(self) -> None:
        """Verify send_state returns False when controller is disconnected."""
        result = self.controller.send_state(STATE_BHAVYA)
        self.assertFalse(result)

    @patch("serial.Serial")
    def test_write_error_disconnects_gracefully(self, mock_serial_cls: MagicMock) -> None:
        """Verify I/O error during write marks connection as disconnected."""
        mock_serial_instance = MagicMock()
        mock_serial_instance.is_open = True
        mock_serial_instance.write.side_effect = serial.SerialException("Device unplugged")
        mock_serial_cls.return_value = mock_serial_instance

        self.controller.connect()
        result = self.controller.send_state(STATE_BHAVYA)

        self.assertFalse(result)
        self.assertFalse(self.controller.is_connected)

    @patch("serial.Serial")
    def test_disconnect_turns_off_leds_and_closes_port(self, mock_serial_cls: MagicMock) -> None:
        """Verify disconnect transmits 'O' before closing port."""
        mock_serial_instance = MagicMock()
        mock_serial_instance.is_open = True
        mock_serial_cls.return_value = mock_serial_instance

        self.controller.connect()
        self.controller.disconnect()

        mock_serial_instance.write.assert_called_with(f"{STATE_IDLE}\n".encode("ascii"))
        mock_serial_instance.close.assert_called_once()
        self.assertFalse(self.controller.is_connected)
        self.assertIsNone(self.controller.last_sent_state)


if __name__ == "__main__":
    unittest.main()
