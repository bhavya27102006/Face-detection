"""Security decision engine and state transition manager.

Implements temporal confirmation and anti-flicker debouncing so single-frame
detection anomalies do not cause rapid state toggling between:
  - 'B' (Bhavya / Authorized -> Green LED)
  - 'U' (Unknown / Unauthorized -> Red LED)
  - 'O' (Nobody / Idle -> LEDs OFF)
"""

import logging
from typing import Optional

from config import (
    CONFIRMATION_FRAMES_BHAVYA,
    CONFIRMATION_FRAMES_NO_FACE,
    CONFIRMATION_FRAMES_UNKNOWN,
    STATE_BHAVYA,
    STATE_IDLE,
    STATE_UNKNOWN,
)

logger = logging.getLogger(__name__)


class SecurityManager:
    """Evaluates face recognition outputs and manages transitions between states (B, U, O)."""

    def __init__(
        self,
        frames_bhavya: int = CONFIRMATION_FRAMES_BHAVYA,
        frames_unknown: int = CONFIRMATION_FRAMES_UNKNOWN,
        frames_no_face: int = CONFIRMATION_FRAMES_NO_FACE,
    ) -> None:
        """Initialize security manager with idle state and anti-flicker thresholds.

        Args:
            frames_bhavya: Consecutive frames required to confirm Bhavya ('B').
            frames_unknown: Consecutive frames required to confirm Unknown ('U').
            frames_no_face: Consecutive frames required to confirm Nobody ('O').
        """
        self.frames_bhavya = frames_bhavya
        self.frames_unknown = frames_unknown
        self.frames_no_face = frames_no_face

        self.current_state: str = STATE_IDLE
        self._candidate_state: Optional[str] = None
        self._candidate_count: int = 0

    @property
    def candidate_state(self) -> Optional[str]:
        """Return the current candidate state being temporally evaluated."""
        return self._candidate_state

    @property
    def candidate_count(self) -> int:
        """Return the number of consecutive occurrences of the current candidate state."""
        return self._candidate_count

    def _map_identity_to_state(self, recognized_identity: str) -> str:
        """Map recognition label to target security state.

        Args:
            recognized_identity: Identity label ('BHAVYA', 'UNKNOWN', 'NO_FACE', etc.)

        Returns:
            str: Target candidate state ('B', 'U', or 'O').
        """
        if recognized_identity == "BHAVYA":
            return STATE_BHAVYA
        elif recognized_identity in ("UNKNOWN", "MULTIPLE_FACES"):
            return STATE_UNKNOWN
        elif recognized_identity == "NO_FACE":
            return STATE_IDLE
        else:
            logger.warning("Unrecognized identity '%s'. Defaulting to idle.", recognized_identity)
            return STATE_IDLE

    def _get_required_frames(self, target_state: str) -> int:
        """Get required consecutive frame count for a given target state."""
        if target_state == STATE_BHAVYA:
            return self.frames_bhavya
        elif target_state == STATE_UNKNOWN:
            return self.frames_unknown
        else:
            return self.frames_no_face

    def evaluate_state(self, recognized_identity: str) -> str:
        """Evaluate raw recognition output with temporal anti-flicker confirmation.

        Args:
            recognized_identity: Raw identity output from FaceRecognizer
                                 ('BHAVYA', 'UNKNOWN', 'NO_FACE', 'MULTIPLE_FACES').

        Returns:
            str: Confirmed system security state ('B', 'U', or 'O').
        """
        target_state = self._map_identity_to_state(recognized_identity)

        # If target matches current confirmed state, reset candidate tracker
        if target_state == self.current_state:
            self._candidate_state = None
            self._candidate_count = 0
            return self.current_state

        # Evaluating transition to a new target state
        if target_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = target_state
            self._candidate_count = 1

        required_frames = self._get_required_frames(target_state)

        if self._candidate_count >= required_frames:
            logger.info(
                "Security state transitioned: '%s' -> '%s' (confirmed after %d frames).",
                self.current_state,
                target_state,
                self._candidate_count,
            )
            self.current_state = target_state
            self._candidate_state = None
            self._candidate_count = 0

        return self.current_state

    def reset(self) -> None:
        """Reset state machine to initial idle state."""
        self.current_state = STATE_IDLE
        self._candidate_state = None
        self._candidate_count = 0
        logger.debug("Security manager state reset to IDLE ('O').")
