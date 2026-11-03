"""
Configuration Module for SERBot Perception & Site Safety Monitoring System (BuildSentinel)
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEMO_DIR = BASE_DIR / "demo_assets"

# Model Filepaths
MODEL_PATHS = {
    "ppe": str(MODELS_DIR / "ppe_yolov8n.pt"),
    "ppe_vest": str(MODELS_DIR / "ppe6_yolov8n.pt"),
    "general": str(MODELS_DIR / "yolov8n.pt"),
    "fire_smoke": str(MODELS_DIR / "fire_smoke_yolov8n.pt"),
    "crack": str(MODELS_DIR / "crack_yolov8n.pt")
}

# Confidence Thresholds
THRESHOLDS = {
    "ppe": 0.35,
    "person": 0.45,
    "obstacle": 0.40,
    "fire_smoke": 0.35,
    "crack": 0.30
}

# Optical Flow / Structural Movement Settings
STRUCTURAL_TRACKING = {
    "max_corners": 150,
    "quality_level": 0.03,
    "min_distance": 15,
    "block_size": 7,
    "displacement_alert_threshold_px": 8.0,  # Alert if movement exceeds 8 pixels
    "tilt_alert_degrees": 2.5               # Alert if structural tilt exceeds 2.5 deg
}

# Distance Estimation Camera Model (Pinhole Approximation)
CAMERA_PARAMS = {
    "focal_length_px": 700.0,   # Nominal focal length for 720p/1080p lens
    "robot_height_m": 0.8,      # Height of SERBot camera above ground
    "tilt_angle_deg": 10.0      # Downward camera tilt
}

# Real-world Typical Heights (meters) for Distance Approximation
REFERENCE_HEIGHTS = {
    "person": 1.75,
    "car": 1.5,
    "truck": 2.8,
    "bus": 3.2,
    "motorcycle": 1.1,
    "bicycle": 1.0,
    "traffic light": 1.0,
    "fire hydrant": 0.8,
    "stop sign": 0.8,
    "bench": 0.8,
    "backpack": 0.5,
    "suitcase": 0.6,
    "chair": 0.85,
    "couch": 0.9,
    "default": 1.0
}

# Collision Proximity Ranges (meters)
PROXIMITY_RANGES = {
    "critical": 1.2,  # Emergency Stop needed
    "warning": 3.0,   # Slow down / Alert path
    "caution": 5.0    # Monitor object
}

# Default Geofenced Unsafe Zones (Polygon normalized coordinates [0.0 - 1.0])
DEFAULT_UNSAFE_ZONES = [
    {
        "id": "zone_excavation",
        "name": "Excavation Pit / Drop Hazard",
        "type": "fall_hazard",
        "color": (0, 0, 255),  # Red
        "polygon": [
            [0.05, 0.65],
            [0.35, 0.65],
            [0.40, 0.95],
            [0.05, 0.95]
        ]
    },
    {
        "id": "zone_crane",
        "name": "Heavy Machinery / Crane Swing",
        "type": "crush_hazard",
        "color": (0, 140, 255),  # Orange
        "polygon": [
            [0.60, 0.40],
            [0.95, 0.40],
            [0.95, 0.85],
            [0.60, 0.85]
        ]
    }
]

# HUD Visual Palette (BGR for OpenCV)
COLOR_PALETTE = {
    "safe": (0, 230, 115),         # Vibrant Neon Green
    "warning": (0, 210, 255),      # Neon Amber / Yellow
    "danger": (40, 60, 255),       # Bright Red
    "critical": (0, 0, 255),       # Pure Red
    "structural": (255, 200, 0),   # Neon Cyan
    "fire": (0, 69, 255),          # Neon Orange-Red
    "person_box": (255, 255, 0),   # Light Cyan
    "hud_bg": (20, 20, 25),        # Deep Charcoal
    "text": (255, 255, 255)        # Crisp White
}
