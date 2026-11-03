"""
Unified SERBot Multi-Hazard Perception Pipeline (BuildSentinel)
Coordinates all AI models, computer vision detectors, and sensor fusion.
"""
import numpy as np
import time
from typing import Dict, Any, Optional
from .config import MODEL_PATHS, THRESHOLDS
from .detectors.ppe_detector import PPEDetector
from .detectors.obstacle_detector import ObstacleDetector
from .detectors.zone_monitor import UnsafeZoneMonitor
from .detectors.structural_monitor import StructuralMonitor
from .detectors.hazard_detector import EnvironmentalHazardDetector
from .sensors.sensor_fusion import SensorFusionEngine
from .visualizer import HUDVisualizer

class SERBotPerceptionEngine:
    """
    Central AI perception engine for SERBot autonomous safety and structural monitoring.
    """
    def __init__(self, custom_model_paths: Optional[Dict[str, str]] = None):
        paths = MODEL_PATHS.copy()
        if custom_model_paths:
            paths.update(custom_model_paths)
            
        print("[SERBot Engine] Initializing Perception Modules...")
        
        # 1. Obstacle & Person Detector (COCO 80 classes)
        self.obstacle_detector = ObstacleDetector(
            model_path=paths["general"],
            conf_threshold=THRESHOLDS["obstacle"]
        )
        
        # 2. PPE Compliance & Safety Gap Detector
        self.ppe_detector = PPEDetector(
            ppe_model_path=paths["ppe"],
            vest_model_path=paths["ppe_vest"],
            conf_threshold=THRESHOLDS["ppe"]
        )
        
        # 3. Unsafe Zone Geofencing Monitor
        self.zone_monitor = UnsafeZoneMonitor()
        
        # 4. Structural Cracks & Movement Tracker
        self.structural_monitor = StructuralMonitor(
            crack_model_path=paths["crack"],
            conf_threshold=THRESHOLDS["crack"]
        )
        
        # 5. Environmental Hazards (Fire, Smoke, Spills)
        self.hazard_detector = EnvironmentalHazardDetector(
            fire_model_path=paths["fire_smoke"],
            conf_threshold=THRESHOLDS["fire_smoke"]
        )
        
        # 6. Sensor Fusion Hub
        self.sensor_fusion = SensorFusionEngine()
        
        # 7. Augmented HUD Visualizer
        self.visualizer = HUDVisualizer()
        
        print("[SERBot Engine] All modules loaded and online!")

    def process_frame(self, frame: np.ndarray, telemetry: Optional[Dict[str, Any]] = None, render_hud: bool = True) -> Dict[str, Any]:
        """
        Executes end-to-end perception pipeline on an image or video frame.
        
        Args:
            frame: BGR numpy image frame
            telemetry: Optional real-time physical sensor data dict
            render_hud: Whether to render visual annotations and HUD overlays
            
        Returns:
            Dict containing annotated frame and comprehensive safety intelligence.
        """
        start_time = time.time()
        
        # Step 1: Detect Obstacles and Persons
        obstacle_res = self.obstacle_detector.detect(frame)
        persons = obstacle_res.get("persons", [])
        
        # Step 2: Detect Safety Gear / PPE Compliance on Workers
        ppe_res = self.ppe_detector.detect(frame, person_boxes=persons)
        
        # Step 3: Monitor Unsafe Zones for Intrusions
        tracked_entities = persons + [obs for obs in obstacle_res["obstacles"] if obs["class"] != "person"]
        zone_res = self.zone_monitor.monitor(frame, tracked_entities)
        
        # Step 4: Structural Health (Cracks & Movement Tracking)
        structural_res = self.structural_monitor.process(frame)
        
        # Step 5: Environmental Hazards (Fire, Smoke, Spills)
        hazard_res = self.hazard_detector.detect(frame)
        
        # Consolidate all visual alerts
        all_alerts = []
        all_alerts.extend(ppe_res.get("violations", []))
        all_alerts.extend(obstacle_res.get("alerts", []))
        all_alerts.extend(zone_res.get("breaches", []))
        all_alerts.extend(structural_res.get("alerts", []))
        all_alerts.extend(hazard_res.get("alerts", []))
        
        # Step 6: Multi-Sensor Telemetry Fusion
        fusion_input = {
            "all_alerts": all_alerts,
            "closest_distance_m": obstacle_res.get("closest_distance_m"),
            "structural": structural_res,
            "ppe": ppe_res
        }
        fusion_res = self.sensor_fusion.fuse(fusion_input, telemetry=telemetry)
        
        # Pipeline execution latency
        latency_ms = round((time.time() - start_time) * 1000.0, 1)
        fps = round(1000.0 / max(1.0, latency_ms), 1)
        
        # Consolidated Perception Report
        report = {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "fps": fps,
            "ppe": ppe_res,
            "obstacles": obstacle_res,
            "zones": zone_res,
            "structural": structural_res,
            "hazards": hazard_res,
            "all_alerts": all_alerts,
            "fusion": fusion_res,
            "total_threats": len(all_alerts) + len(fusion_res.get("fused_alerts", []))
        }
        
        # Step 7: Render Visual Annotations and HUD
        if render_hud:
            annotated_frame = self.visualizer.render(frame, report, show_zones=True, show_hud=True)
            report["annotated_frame"] = annotated_frame
        else:
            report["annotated_frame"] = frame
            
        return report
