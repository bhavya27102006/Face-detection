"""Serial communication controller for ESP-12E microcontroller.

Handles USB-serial connectivity, state transmission ('B', 'U', 'O'),
error handling, and duplicate command suppression.
"""

import logging
from typing import Optional
import serial

from config import (
    BAUD_RATE,
    SERIAL_PORT,
    SERIAL_TIMEOUT,
    STATE_IDLE,
    VALID_STATES,
)

logger = logging.getLogger(__name__)


class ESPController:
    """Manages serial connection and sends state commands ('B', 'U', 'O') to ESP-12E."""

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud_rate: int = BAUD_RATE,
        timeout: float = SERIAL_TIMEOUT,
    ) -> None:
        """Initialize serial connection parameters.

        Args:
            port: Serial port identifier (e.g. 'COM5').
            baud_rate: Baud rate for communication (default: 115200).
            timeout: Read/write timeout in seconds.
        """
        self.port: str = port
        self.baud_rate: int = baud_rate
        self.timeout: float = timeout
        self.connection: Optional[serial.Serial] = None
        self._last_sent_state: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Return True if serial connection is open and active."""
        return self.connection is not None and self.connection.is_open

    @property
    def last_sent_state(self) -> Optional[str]:
        """Return the most recently transmitted state."""
        return self._last_sent_state

    def connect(self) -> bool:
        """Open USB serial connection with ESP-12E.

        Returns:
            bool: True if connection succeeded, False otherwise.
        """
        if self.is_connected:
            logger.info("ESP-12E already connected on %s.", self.port)
            return True

        try:
            logger.info(
                "Connecting to ESP-12E on port %s at %d baud...",
                self.port,
                self.baud_rate,
            )
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            # Reset buffer to purge old/partial data
            self.connection.reset_input_buffer()
            self.connection.reset_output_buffer()
            self._last_sent_state = None
            logger.info("Successfully connected to ESP-12E on %s.", self.port)
            return True
        except (serial.SerialException, OSError) as err:
            logger.error("Failed to connect to ESP-12E on %s: %s", self.port, err)
            self.connection = None
            return False

    def send_state(self, state: str, force: bool = False) -> bool:
        """Transmit state character ('B', 'U', or 'O') to ESP-12E.

        Args:
            state: System state command ('B', 'U', or 'O').
            force: If True, send even if state matches last_sent_state.

        Returns:
            bool: True if transmitted successfully or duplicate ignored, False on error.
        """
        if state not in VALID_STATES:
            logger.warning(
                "Invalid state '%s'. Must be one of %s.",
                state,
                VALID_STATES,
            )
            return False

        if not self.is_connected or self.connection is None:
            logger.warning(
                "Cannot send state '%s': ESP-12E is not connected.",
                state,
            )
            return False

        if not force and state == self._last_sent_state:
            # Duplicate state, no need to flood the serial link
            return True

        try:
            payload = f"{state}\n".encode("ascii")
            self.connection.write(payload)
            self.connection.flush()
            self._last_sent_state = state
            logger.info("Sent state '%s' to ESP-12E.", state)
            return True
        except (serial.SerialException, OSError) as err:
            logger.error("Error sending state '%s' to ESP-12E: %s", state, err)
            # Mark disconnected on I/O failure
            self.disconnect()
            return False

    def disconnect(self) -> None:
        """Close USB serial connection cleanly and safely turn off LEDs."""
        if self.connection is not None:
            try:
                if self.connection.is_open:
                    # Turn off LEDs before shutting down connection
                    try:
                        self.connection.write(f"{STATE_IDLE}\n".encode("ascii"))
                        self.connection.flush()
                    except (serial.SerialException, OSError):
                        pass
                    self.connection.close()
                    logger.info("Disconnected from ESP-12E on %s.", self.port)
            except Exception as err:
                logger.warning("Error during serial disconnect: %s", err)
            finally:
                self.connection = None
                self._last_sent_state = None
