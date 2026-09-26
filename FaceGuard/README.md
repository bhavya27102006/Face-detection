# FaceGuard

FaceGuard is a real-time face monitoring and hardware security alert system using a laptop webcam, Python, and an ESP-12E microcontroller.

## System States & Hardware Mapping

| Command | State Description | ESP-12E Pin | Output Indicator |
| :--- | :--- | :--- | :--- |
| `B` | Bhavya detected | `D0` (GPIO16) | Green LED ON |
| `U` | Unknown person detected | `D1` (GPIO5) | Red LED ON |
| `O` | Nobody detected / Idle | All | LEDs OFF |

## Project Structure

```text
FaceGuard/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── camera/
│   ├── __init__.py
│   └── camera.py
│
├── recognition/
│   ├── __init__.py
│   └── face_recognition.py
│
├── security/
│   ├── __init__.py
│   └── security_manager.py
│
├── esp/
│   ├── __init__.py
│   └── esp_controller.py
│
├── data/
│   └── .gitkeep
│
├── logs/
│   └── .gitkeep
│
└── tests/
    └── .gitkeep
```
