"""FaceGuard - Real-time Webcam Face Security & ESP-12E Hardware Alert System.

Main application entry point orchestrating:
  CameraStream -> FaceRecognizer -> SecurityManager -> ESPController

Usage:
  python main.py           # Run continuous real-time security monitoring
  python main.py --enroll  # Run interactive face enrollment for Bhavya
"""

import argparse
import logging
import sys
import time
from typing import Optional

from camera.camera import CameraStream
from config import (
    BAUD_RATE,
    ENROLLMENT_SAMPLE_COUNT,
    SERIAL_PORT,
    STATE_BHAVYA,
    STATE_IDLE,
    STATE_UNKNOWN,
)
from esp.esp_controller import ESPController
from recognition.face_recognition import FaceRecognizer
from security.security_manager import SecurityManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FaceGuard")


def run_enrollment(samples_needed: int = ENROLLMENT_SAMPLE_COUNT) -> bool:
    """Capture face samples from webcam and enroll Bhavya's facial embeddings.

    Args:
        samples_needed: Number of valid face frames required.

    Returns:
        bool: True if enrollment succeeded, False otherwise.
    """
    print("\n" + "=" * 55)
    print("   FACEGUARD — BHAVYA FACE ENROLLMENT")
    print("=" * 55)
    print(f"Please sit directly in front of the laptop webcam.")
    print(f"Collecting {samples_needed} clean facial samples...")

    recognizer = FaceRecognizer()
    camera = CameraStream()

    if not camera.start():
        print("Error: Could not access webcam for enrollment.")
        return False

    collected_embeddings = []
    attempts = 0
    max_attempts = 300  # ~20 seconds window at 15 FPS

    try:
        # Allow webcam sensor exposure to stabilize
        time.sleep(1.0)

        while len(collected_embeddings) < samples_needed and attempts < max_attempts:
            attempts += 1
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            detection_result = recognizer.detector.detect(frame)
            if detection_result.is_single_face:
                face = detection_result.faces[0]
                if face.confidence >= 0.65:
                    feat = recognizer.extract_features(frame, face)
                    if feat is not None:
                        collected_embeddings.append(feat)
                        print(f"  [+] Captured sample {len(collected_embeddings)}/{samples_needed} "
                              f"(confidence: {face.confidence:.2f})")
                        time.sleep(0.3)
            elif detection_result.is_multiple_faces:
                print("  [!] Multiple faces detected. Please ensure only one person is in frame.")
                time.sleep(0.5)

            time.sleep(0.05)

        if len(collected_embeddings) < samples_needed:
            print("Error: Enrollment timed out before capturing enough face samples.")
            return False

        success = recognizer.enroll_face(collected_embeddings)
        if success:
            print("\nEnrollment successfully completed and saved locally.")
            return True
        else:
            print("Error: Failed to save reference embeddings to disk.")
            return False

    finally:
        camera.stop()


def run_security_monitor(max_cycles: Optional[int] = None) -> None:
    """Run the continuous FaceGuard security monitoring loop.

    Args:
        max_cycles: Optional integer to limit loop iterations (useful for testing).
    """
    print("\n" + "=" * 55)
    print("   FACEGUARD — ACTIVE SECURITY MONITOR")
    print("=" * 55)

    # Initialize components
    esp = ESPController(port=SERIAL_PORT, baud_rate=BAUD_RATE)
    camera = CameraStream()
    recognizer = FaceRecognizer()
    security = SecurityManager()

    # Hardware connection
    print(f"Connecting to ESP-12E on {SERIAL_PORT}...")
    esp_connected = esp.connect()
    if esp_connected:
        print(f"ESP-12E connected. LEDs ready.")
    else:
        print(f"Warning: ESP-12E could not be opened on {SERIAL_PORT}. Running in software-only mode.")

    # Enrollment check
    if not recognizer.is_enrolled:
        print("\nWarning: No enrolled reference face found for Bhavya.")
        print("All detected persons will be treated as UNKNOWN ('U').")
        print("Run 'python main.py --enroll' to register Bhavya's face.\n")

    # Start webcam stream
    print("Starting webcam stream...")
    if not camera.start():
        print("Fatal Error: Could not initialize camera stream.")
        if esp_connected:
            esp.disconnect()
        return

    print("\nFaceGuard is running continuously. Press Ctrl+C to terminate.")
    print("-" * 55)

    cycles = 0
    state_descriptions = {
        STATE_BHAVYA: "BHAVYA (Authorized) -> Green LED ON",
        STATE_UNKNOWN: "UNKNOWN PERSON (Intruder) -> Red LED ON",
        STATE_IDLE: "NOBODY (Idle) -> LEDs OFF",
    }

    try:
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # 1. Face Recognition
            identity, confidence = recognizer.identify_face(frame)

            # 2. Security Decision with Temporal Anti-Flicker
            confirmed_state = security.evaluate_state(identity)

            # 3. Transmit to ESP-12E (duplicate states are automatically filtered)
            if esp_connected:
                esp.send_state(confirmed_state)

            # Log status when confirmed state changes or periodically
            desc = state_descriptions.get(confirmed_state, confirmed_state)
            sys.stdout.write(
                f"\r[{cycles:5d}] Detected: {identity:<14} | Confirmed: {confirmed_state} ({desc:<35})"
            )
            sys.stdout.flush()

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n\nShutting down FaceGuard on user request...")

    finally:
        print("\nCleaning up resources...")
        camera.stop()
        if esp_connected:
            esp.disconnect()
        print("FaceGuard shutdown complete. Goodbye.\n")


def main() -> None:
    """Parse command line arguments and execute FaceGuard."""
    parser = argparse.ArgumentParser(description="FaceGuard - Webcam Face Recognition & Hardware Alert System")
    parser.add_argument("--enroll", action="store_true", help="Enroll Bhavya's reference face data")
    parser.add_argument("--test-cycles", type=int, default=None, help="Run monitor for N cycles (testing)")
    args = parser.parse_args()

    if args.enroll:
        success = run_enrollment()
        sys.exit(0 if success else 1)
    else:
        run_security_monitor(max_cycles=args.test_cycles)


if __name__ == "__main__":
    main()
