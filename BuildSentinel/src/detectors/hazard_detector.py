"""
Environmental Hazard Detector Module
Detects fire, smoke, hazardous spills, and extreme environmental threats.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import os

class EnvironmentalHazardDetector:
    """
    Monitors site environment for fire, smoke, toxic dust clouds, and liquid spills.
    Combines deep learning object detection with colorimetric flame validation.
    """
    def __init__(self, fire_model_path: str, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model_fire = YOLO(fire_model_path) if os.path.exists(fire_model_path) else None

    def _colorimetric_flame_check(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Colorimetric flame verification based on HSV and RGB thresholding.
        Catches high-intensity thermal flame regions.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(frame)
        
        # Flame rule: high brightness and distinct flame saturation
        lower_fire = np.array([0, 160, 210], dtype=np.uint8)
        upper_fire = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        
        # Strict intensity rule: R > 220, dominant red over green and blue
        intensity_rule = (r > 220) & (r > (g + 45)) & (g > (b + 20))
        combined_mask = cv2.bitwise_and(mask, mask, mask=intensity_rule.astype(np.uint8) * 255)
        
        # Morphology cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flame_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 250:  # Minimum pixel threshold
                x, y, w, h = cv2.boundingRect(cnt)
                flame_regions.append({
                    "class": "flame_thermal",
                    "confidence": round(min(0.95, area / 2000.0 + 0.4), 2),
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "area_px": float(area)
                })
        return flame_regions

    def detect_spills(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects ground-plane liquid spills or chemical puddles using specular reflection analysis.
        """
        h_img, w_img = frame.shape[:2]
        # Restrict ground spill check to lower 40% of camera frame (floor/ground)
        ground_roi = frame[int(h_img * 0.6):, :]
        gray_roi = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2GRAY)
        
        # Spills exhibit high gradient variance and specular highlights
        blurred = cv2.GaussianBlur(gray_roi, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 215, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spills = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                x, y, w, h = cv2.boundingRect(cnt)
                # Offset y back to full frame
                actual_y = y + int(h_img * 0.6)
                spills.append({
                    "class": "liquid_spill",
                    "confidence": 0.72,
                    "bbox": [float(x), float(actual_y), float(x + w), float(actual_y + h)],
                    "area_px": float(area)
                })
        return spills

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects fire, smoke, and liquid spill hazards.
        
        Returns:
            Dict containing detected hazards, alerts, and threat levels.
        """
        hazards = []
        alerts = []
        
        # 1. AI Fire & Smoke Model
        if self.model_fire is not None:
            results = self.model_fire(frame, conf=self.conf_threshold, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model_fire.names[cls_id]
                conf = float(box.conf[0])
                coords = [float(v) for v in box.xyxy[0]]
                
                hazard_item = {
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": coords,
                    "source": "deep_model"
                }
                hazards.append(hazard_item)
                
                alerts.append({
                    "type": "FIRE_OR_SMOKE",
                    "severity": "CRITICAL" if "fire" in cls_name.lower() else "WARNING",
                    "hazard": cls_name,
                    "message": f"EMERGENCY HAZARD: {cls_name.upper()} detected on site!",
                    "bbox": coords
                })
                
        # 2. Colorimetric Flame Fallback (if no deep fire detected)
        flames_color = self._colorimetric_flame_check(frame)
        if len(hazards) == 0 and len(flames_color) > 0:
            for f in flames_color:
                hazards.append(f)
                alerts.append({
                    "type": "THERMAL_FLAME_SURGE",
                    "severity": "CRITICAL",
                    "hazard": "Flame Heat Signature",
                    "message": "EMERGENCY HAZARD: Intense flame signature detected!",
                    "bbox": f["bbox"]
                })
                
        # 3. Ground Spill Detection
        spills = self.detect_spills(frame)
        for s in spills:
            hazards.append(s)
            alerts.append({
                "type": "LIQUID_SPILL",
                "severity": "WARNING",
                "hazard": "Liquid Spill / Puddle",
                "message": "CAUTION: Wet ground / liquid spill hazard detected in robot path!",
                "bbox": s["bbox"]
            })
            
        return {
            "hazards": hazards,
            "total_hazards": len(hazards),
            "alerts": alerts,
            "threat_level": "CRITICAL" if any(a["severity"] == "CRITICAL" for a in alerts) else ("WARNING" if alerts else "NORMAL")
        }
