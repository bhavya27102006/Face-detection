"""Unit tests for SecurityManager module and temporal anti-flicker state machine."""

import unittest
from security.security_manager import SecurityManager
from config import STATE_BHAVYA, STATE_UNKNOWN, STATE_IDLE


class TestSecurityManager(unittest.TestCase):
    """Test suite for SecurityManager state machine and temporal debouncing."""

    def setUp(self) -> None:
        """Create fresh SecurityManager with default thresholds (3 Bhavya, 4 Unknown, 3 Idle)."""
        self.sm = SecurityManager(frames_bhavya=3, frames_unknown=4, frames_no_face=3)

    def test_initial_state_is_idle(self) -> None:
        """Verify manager initializes in idle state ('O')."""
        self.assertEqual(self.sm.current_state, STATE_IDLE)
        self.assertIsNone(self.sm.candidate_state)
        self.assertEqual(self.sm.candidate_count, 0)

    def test_transition_to_bhavya_requires_consecutive_frames(self) -> None:
        """Verify Bhavya ('B') requires 3 consecutive positive frames."""
        # Frame 1
        s1 = self.sm.evaluate_state("BHAVYA")
        self.assertEqual(s1, STATE_IDLE)
        self.assertEqual(self.sm.candidate_state, STATE_BHAVYA)
        self.assertEqual(self.sm.candidate_count, 1)

        # Frame 2
        s2 = self.sm.evaluate_state("BHAVYA")
        self.assertEqual(s2, STATE_IDLE)
        self.assertEqual(self.sm.candidate_count, 2)

        # Frame 3 - Confirmed!
        s3 = self.sm.evaluate_state("BHAVYA")
        self.assertEqual(s3, STATE_BHAVYA)
        self.assertEqual(self.sm.current_state, STATE_BHAVYA)

    def test_spurious_single_frame_glitch_does_not_flicker_state(self) -> None:
        """Verify single frame noise does not trigger a false state switch."""
        # Establish Bhavya state
        for _ in range(3):
            self.sm.evaluate_state("BHAVYA")
        self.assertEqual(self.sm.current_state, STATE_BHAVYA)

        # Single frame of UNKNOWN (e.g. motion blur or turn)
        glitch_state = self.sm.evaluate_state("UNKNOWN")
        # Confirmed state must remain BHAVYA
        self.assertEqual(glitch_state, STATE_BHAVYA)
        self.assertEqual(self.sm.candidate_state, STATE_UNKNOWN)
        self.assertEqual(self.sm.candidate_count, 1)

        # Next frame recovers to BHAVYA
        recovered = self.sm.evaluate_state("BHAVYA")
        self.assertEqual(recovered, STATE_BHAVYA)
        # Candidate tracker should be cleared
        self.assertIsNone(self.sm.candidate_state)
        self.assertEqual(self.sm.candidate_count, 0)

    def test_transition_to_unknown_requires_confirmation(self) -> None:
        """Verify Unknown ('U') requires 4 consecutive frames before triggering alert."""
        for i in range(3):
            st = self.sm.evaluate_state("UNKNOWN")
            self.assertEqual(st, STATE_IDLE)

        # 4th frame triggers U
        st4 = self.sm.evaluate_state("UNKNOWN")
        self.assertEqual(st4, STATE_UNKNOWN)

    def test_multiple_faces_treated_as_unknown(self) -> None:
        """Verify MULTIPLE_FACES maps to candidate 'U'."""
        for _ in range(4):
            st = self.sm.evaluate_state("MULTIPLE_FACES")
        self.assertEqual(st, STATE_UNKNOWN)

    def test_full_continuous_person_transition_scenario(self) -> None:
        """Verify the complete user lifecycle:
        1. Bhavya arrives -> Green ('B')
        2. Bhavya leaves -> Idle ('O')
        3. Unknown arrives -> Red ('U')
        4. Unknown leaves -> Idle ('O')
        5. Bhavya returns -> Green ('B')
        """
        # 1. Bhavya arrives
        for _ in range(3):
            st = self.sm.evaluate_state("BHAVYA")
        self.assertEqual(st, STATE_BHAVYA)

        # 2. Bhavya leaves
        for _ in range(3):
            st = self.sm.evaluate_state("NO_FACE")
        self.assertEqual(st, STATE_IDLE)

        # 3. Unknown enters
        for _ in range(4):
            st = self.sm.evaluate_state("UNKNOWN")
        self.assertEqual(st, STATE_UNKNOWN)

        # 4. Unknown leaves
        for _ in range(3):
            st = self.sm.evaluate_state("NO_FACE")
        self.assertEqual(st, STATE_IDLE)

        # 5. Bhavya returns
        for _ in range(3):
            st = self.sm.evaluate_state("BHAVYA")
        self.assertEqual(st, STATE_BHAVYA)

    def test_reset(self) -> None:
        """Verify reset clears state back to IDLE."""
        for _ in range(3):
            self.sm.evaluate_state("BHAVYA")
        self.assertEqual(self.sm.current_state, STATE_BHAVYA)

        self.sm.reset()
        self.assertEqual(self.sm.current_state, STATE_IDLE)


if __name__ == "__main__":
    unittest.main()
