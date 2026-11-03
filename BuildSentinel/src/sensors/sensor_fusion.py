"""
Sensor Fusion & SERBot Hardware Telemetry Module
Fuses Computer Vision alerts with physical sensor streams:
- Ultrasonic / LiDAR Distance
- MQ-2 / MQ-135 Gas & Smoke Sensors
- MPU-6050 IMU Accelerometer / Vibration
- DHT-22 Ambient Temperature & Humidity
"""
import time
import random
from typing import Dict, Any, List

class SensorFusionEngine:
    """
    Integrates multi-sensor telemetry with computer vision detections
    to provide robust, cross-verified safety assessments for SERBot.
    """
    def __init__(self):
        # Default baseline sensor readings
        self.last_telemetry = {
            "timestamp": time.time(),
            "ultrasonic_distance_cm": 250.0,
            "gas_smoke_ppm": 35.0,           # Baseline clean air is ~30-50 ppm
            "temperature_c": 28.5,
            "humidity_pct": 52.0,
            "vibration_g": 0.08,             # Baseline floor vibration
            "battery_pct": 94.0
        }

    def update_hardware_telemetry(self, telemetry: Dict[str, Any]):
        """Injects live sensor data from SERBot Arduino/Raspberry Pi/ESP32."""
        self.last_telemetry.update(telemetry)
        self.last_telemetry["timestamp"] = time.time()

    def simulate_telemetry(self, vision_alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulates realistic sensor responses correlated with visual detections
        for testing, simulation, and offline demonstrations.
        """
        # Baseline noise
        t_temp = 28.0 + random.uniform(-0.5, 0.5)
        t_gas = 40.0 + random.uniform(-3.0, 3.0)
        t_vib = 0.08 + random.uniform(-0.02, 0.02)
        t_dist = 260.0 + random.uniform(-5.0, 5.0)
        
        # Check if vision detected hazards
        has_fire = any(a.get("type") in ["FIRE_OR_SMOKE", "THERMAL_FLAME_SURGE"] for a in vision_alerts)
        has_structural = any(a.get("type") in ["STRUCTURAL_SHIFT", "STRUCTURAL_DEFECT"] for a in vision_alerts)
        has_obstacle = any(a.get("type") == "OBSTACLE_COLLISION_RISK" for a in vision_alerts)
        
        if has_fire:
            t_gas = random.uniform(320.0, 580.0)      # High smoke/gas
            t_temp = random.uniform(55.0, 82.0)       # Elevated heat
            
        if has_structural:
            t_vib = random.uniform(1.2, 2.8)          # High structural acceleration
            
        if has_obstacle:
            # Correlate distance
            t_dist = random.uniform(40.0, 110.0)      # Close obstacle in cm
            
        telemetry = {
            "timestamp": round(time.time(), 2),
            "ultrasonic_distance_cm": round(t_dist, 1),
            "gas_smoke_ppm": round(t_gas, 1),
            "temperature_c": round(t_temp, 1),
            "humidity_pct": round(52.0 + random.uniform(-2, 2), 1),
            "vibration_g": round(t_vib, 2),
            "battery_pct": round(max(10.0, self.last_telemetry.get("battery_pct", 94.0) - 0.01), 1)
        }
        self.last_telemetry = telemetry
        return telemetry

    def fuse(self, vision_results: Dict[str, Any], telemetry: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Performs multi-modal sensor fusion between Computer Vision and physical sensor telemetry.
        
        Returns:
            Dict containing composite site safety score, fused emergency alerts,
            and SERBot robotic actuation directives.
        """
        if telemetry is None:
            telemetry = self.simulate_telemetry(vision_results.get("all_alerts", []))
            
        fused_alerts = []
        vision_alerts = vision_results.get("all_alerts", [])
        
        # 1. Fire / Hazard Cross-Verification
        vision_fire = any(a.get("type") in ["FIRE_OR_SMOKE", "THERMAL_FLAME_SURGE"] for a in vision_alerts)
        gas_high = telemetry.get("gas_smoke_ppm", 0) > 200.0
        temp_high = telemetry.get("temperature_c", 0) > 50.0
        
        if vision_fire and gas_high:
            fused_alerts.append({
                "type": "CONFIRMED_FIRE_DISASTER",
                "severity": "CRITICAL",
                "message": f"DUAL CONFIRMATION: Vision & Gas Sensor ({telemetry['gas_smoke_ppm']} ppm) confirm active fire!",
                "directive": "TRIGGER_ALARM_AND_EVACUATE"
            })
        elif gas_high:
            fused_alerts.append({
                "type": "TOXIC_GAS_LEAK",
                "severity": "WARNING",
                "message": f"Sensor Alert: Elevated gas/combustion particles ({telemetry['gas_smoke_ppm']} ppm)!",
                "directive": "DISPATCH_AIR_SAMPLING_BOT"
            })
            
        # 2. Obstacle & Ultrasonic Distance Cross-Verification
        vision_dist = vision_results.get("closest_distance_m")
        sensor_dist_m = telemetry.get("ultrasonic_distance_cm", 999.0) / 100.0
        
        if (vision_dist and vision_dist < 1.2) or (sensor_dist_m < 1.0):
            fused_alerts.append({
                "type": "CONFIRMED_COLLISION_HAZARD",
                "severity": "CRITICAL",
                "message": f"Proximity Alert: Object within {min(filter(None, [vision_dist, sensor_dist_m])):.2f}m!",
                "directive": "EMERGENCY_BRAKE_ACTIVE"
            })
            
        # 3. Structural Movement & Vibration Sensor Fusion
        structural_movement = vision_results.get("structural", {}).get("movement", {}).get("movement_detected", False)
        high_vib = telemetry.get("vibration_g", 0) > 1.0
        
        if structural_movement and high_vib:
            fused_alerts.append({
                "type": "CONFIRMED_STRUCTURAL_INSTABILITY",
                "severity": "CRITICAL",
                "message": f"SEISMIC/STRUCTURAL SHIFT: Vision displacement + {telemetry['vibration_g']}g IMU oscillation!",
                "directive": "CORDON_OFF_SECTOR"
            })
            
        # 4. Compute Overall Safety Index (0 to 100)
        penalty = 0
        for alert in vision_alerts + fused_alerts:
            sev = alert.get("severity", "INFO")
            if sev == "CRITICAL":
                penalty += 35
            elif sev == "WARNING":
                penalty += 15
            elif sev == "CAUTION":
                penalty += 5
                
        # Worker PPE compliance factor
        compliance_rate = vision_results.get("ppe", {}).get("compliance_rate", 100.0)
        safety_score = max(0, min(100, int((compliance_rate * 0.4) + (100 - penalty) * 0.6)))
        
        if safety_score >= 80:
            status = "OPTIMAL_SAFE"
        elif safety_score >= 50:
            status = "ELEVATED_RISK"
        else:
            status = "CRITICAL_HAZARD"
            
        return {
            "telemetry": telemetry,
            "fused_alerts": fused_alerts,
            "site_safety_score": safety_score,
            "overall_status": status
        }
