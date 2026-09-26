"""FaceGuard Configuration.

Contains global constants and configuration settings for camera,
recognition thresholds, hardware communication, and system states.
"""

# Hardware Serial Configuration (ESP-12E)
SERIAL_PORT: str = "COM5"          # ESP-12E USB-SERIAL CH340 COM port
BAUD_RATE: int = 115200            # Serial communication speed
SERIAL_TIMEOUT: float = 1.0        # Read/write timeout in seconds

# Camera Configuration
CAMERA_INDEX: int = 0              # Default webcam index
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

# Face Detection Configuration (OpenCV YuNet)
FACE_DETECTION_MODEL_PATH: str = "data/models/face_detection_yunet_2023mar.onnx"
FACE_SCORE_THRESHOLD: float = 0.6  # Detection confidence threshold
FACE_NMS_THRESHOLD: float = 0.3    # Non-maximum suppression threshold

# Face Recognition Configuration (OpenCV SFace)
FACE_RECOGNITION_MODEL_PATH: str = "data/models/face_recognition_sface_2021dec.onnx"
ENROLLMENT_DIR: str = "data/known_faces"
BHAVYA_ENROLLMENT_FILE: str = "data/known_faces/bhavya.npy"
RECOGNITION_COSINE_THRESHOLD: float = 0.38  # SFace cosine similarity threshold (>= 0.38 is match)
ENROLLMENT_SAMPLE_COUNT: int = 5            # Number of clean samples to capture during enrollment

# Security States sent to ESP-12E
STATE_BHAVYA: str = "B"            # Authorized user (Bhavya) -> D0 (GPIO16) Green LED ON
STATE_UNKNOWN: str = "U"           # Unknown person detected -> D1 (GPIO5) Red LED ON
STATE_IDLE: str = "O"              # Nobody detected / idle -> All LEDs OFF
VALID_STATES: tuple[str, ...] = (STATE_BHAVYA, STATE_UNKNOWN, STATE_IDLE)

# Temporal Confirmation / Anti-Flicker Thresholds (consecutive frames needed to confirm state)
CONFIRMATION_FRAMES_BHAVYA: int = 3   # 3 consecutive positive frames to switch to Green
CONFIRMATION_FRAMES_UNKNOWN: int = 4  # 4 consecutive unknown frames to switch to Red (prevents blur false alarms)
CONFIRMATION_FRAMES_NO_FACE: int = 3  # 3 consecutive empty frames to turn LEDs OFF
