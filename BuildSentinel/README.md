# SERBot BuildSentinel 🛡️🤖
### Autonomous AI Vision, Site Safety & Structural Health Monitoring Engine

**BuildSentinel** is a computer vision and sensor fusion perception system engineered for **SERBot** (Sensor-Based Robotics). It provides real-time site safety evaluation, PPE compliance monitoring, obstacle detection with distance estimation, dynamic geofencing, structural movement and crack analysis, and multi-hazard environmental perception.

---

## 🚀 Key Perception Capabilities

### 1. Safety Gear & PPE Compliance Detection ("Gaps in Safety")
- **Dual-Model Ensemble**:
  - `ppe_yolov8n.pt`: 10 classes (`helmet`, `no_helmet`, `glove`, `no_glove`, `goggles`, `no_goggles`, `mask`, `no_mask`, `shoes`, `no_shoes`).
  - `ppe6_yolov8n.pt`: High-Visibility Safety Vest, Helmet, Gloves, Goggles, and Safety Shoes.
- **Worker Hierarchy & Spatial Containment**:
  - Automatically identifies workers on site and evaluates their personal protective equipment.
  - Flags explicit non-compliance: e.g., `Worker #1 Missing Hardhat & Safety Vest`.
  - Calculates real-time **PPE Compliance Rate (%)**.

### 2. Obstacle Detection & Autonomous Navigation Directives
- **COCO 80-Class General Obstacle Detector (`yolov8n.pt`)**:
  - Detects heavy machinery, vehicles, trucks, scaffolding, construction containers, barrels, debris, and workers.
- **Pinhole Camera Distance & Bearing Estimation**:
  - Computes real-world distance in meters ($d \approx \frac{f \cdot H_{real}}{h_{bbox}}$) and azimuth angle (LEFT, CENTER, RIGHT).
- **SERBot Robotic Directives**:
  - `CRITICAL (< 1.2m)`: Triggers `EMERGENCY_STOP`.
  - `WARNING (1.2m - 3.0m)`: Issues steering advisories (`STEER_LEFT` or `STEER_RIGHT`).
  - `SAFE (> 3.0m)`: `PROCEED_SAFE`.

### 3. Dynamic Geofenced Unsafe Zones
- Monitors hazardous virtual zones:
  - Excavation pits / drop hazards
  - Crane swing radiuses / heavy equipment operational corridors
  - High-voltage / transformer zones
- Performs sub-pixel point-in-polygon tests on ground contact points.
- Triggers instant `ZONE_INTRUSION` breach alerts with penetration depth.

### 4. Structural Health & Movement Tracking
- **Surface Crack Detection (`crack_yolov8n.pt`)**:
  - Identifies concrete, masonry, and steel cracks.
  - Computes bounding dimensions, edge severity, and pixel length.
- **Optical Flow Structural Displacement Tracker**:
  - Tracks structural keypoints (columns, load-bearing joints, retaining walls) across consecutive camera frames using Lucas-Kanade optical flow.
  - Measures displacement vector $(\Delta x, \Delta y)$ and alerts on micromovements or structural tilt deflections exceeding tolerance thresholds.

### 5. Environmental Hazard Detection & Sensor Fusion
- **Visual Fire & Smoke Perception (`fire_smoke_yolov8n.pt`)**:
  - Detects active flames, combustion plumes, and thermal signatures.
- **Ground Liquid Spill Detection**:
  - Identifies puddles, chemical leaks, and slip hazards via specular reflection analysis.
- **SERBot Hardware Sensor Fusion**:
  - Ingests telemetry streams: MQ-2 Gas/Smoke (ppm), Ultrasonic/LiDAR range (cm), MPU-6050 IMU vibration ($g$), DHT-22 Temperature (°C).
  - Cross-verifies vision with physical sensors (e.g., Vision Fire + Gas > 200 ppm = **Confirmed Disaster Alert**).
  - Generates a composite **Site Safety Index (0 - 100)**.

---

## 📂 Project Architecture

```
BuildSentinel/
├── models/
│   ├── yolov8n.pt              # General obstacles, vehicles, persons (COCO)
│   ├── ppe_yolov8n.pt          # Negative & positive PPE (helmet/no_helmet, gloves, etc.)
│   ├── ppe6_yolov8n.pt         # High-Vis safety vest, boots, gear
│   ├── fire_smoke_yolov8n.pt   # Fire and smoke detection
│   └── crack_yolov8n.pt        # Structural crack detection
├── src/
│   ├── config.py               # Thresholds, camera geometry, geofence polygons
│   ├── visualizer.py           # Futuristic augmented HUD renderer
│   ├── pipeline.py             # Master SERBot Perception Engine
│   ├── detectors/
│   │   ├── ppe_detector.py      # PPE compliance & gap analysis
│   │   ├── obstacle_detector.py # Obstacle classification, distance & bearing
│   │   ├── zone_monitor.py      # Virtual polygon geofencing
│   │   ├── structural_monitor.py# Crack detection + Optical flow displacement
│   │   └── hazard_detector.py   # Fire, smoke, and liquid spill detection
│   └── sensors/
│       └── sensor_fusion.py     # Hardware telemetry fusion & safety scoring
├── web/
│   ├── static/css/style.css    # Premium glassmorphic industrial dark UI
│   ├── static/js/app.js        # Live streaming & reactive telemetry dashboard
│   └── templates/index.html    # Full-featured site safety monitoring portal
├── demo_assets/                # Ready-to-run site inspection test imagery
├── app.py                      # Flask live streaming server & REST API
├── demo.py                     # Standalone CLI runner (image / video / webcam)
├── test_system.py              # Automated system test suite
└── requirements.txt            # Python dependencies
```

---

## ⚡ Quickstart

### 1. Run Diagnostic Self-Test
Verify all 5 perception modules and sensor fusion:
```bash
python test_system.py
```

### 2. Run CLI Inspection
Inspect any site image or test scenario:
```bash
python demo.py --image demo_assets/construction_workers.jpg
python demo.py --image demo_assets/heavy_machinery.jpg
python demo.py --webcam
```

### 3. Launch the Web Dashboard
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser to:
- View live streaming video with real-time HUD annotations.
- Upload any site inspection photo or click pre-loaded test scenarios.
- Monitor PPE compliance, obstacle radar, structural shifts, and sensor telemetry.
- Export full JSON safety audit reports.

---

## 🛠️ Python Integration for SERBot

```python
import cv2
from src.pipeline import SERBotPerceptionEngine

# 1. Initialize perception engine (all 5 models load automatically)
engine = SERBotPerceptionEngine()

# 2. Ingest camera frame from robot
frame = cv2.imread("demo_assets/construction_workers.jpg")

# 3. Optional: Ingest physical sensor telemetry from Arduino / ESP32
telemetry = {
    "gas_smoke_ppm": 42.0,
    "temperature_c": 29.1,
    "ultrasonic_distance_cm": 180.0,
    "vibration_g": 0.12
}

# 4. Process frame
report = engine.process_frame(frame, telemetry=telemetry, render_hud=True)

# 5. Access intelligent insights
print("Safety Index :", report["fusion"]["site_safety_score"])
print("Nav Directive:", report["obstacles"]["nav_command"])
print("Workers PPE  :", report["ppe"]["compliance_rate"], "% compliant")
print("Active Alerts:", len(report["all_alerts"]))

# 6. Save or stream HUD visual
cv2.imwrite("output_hud.jpg", report["annotated_frame"])
```
