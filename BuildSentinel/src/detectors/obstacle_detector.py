"""
Obstacle Detection & Proximity Alert Module for Autonomous Robot Navigation
Estimates distance, bearing, and collision hazard levels for on-site obstacles.
"""
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import os
from ..config import REFERENCE_HEIGHTS, CAMERA_PARAMS, PROXIMITY_RANGES

class ObstacleDetector:
    """
    Detects dynamic and static obstacles, estimates real-world distance,
    and generates robotic navigation recommendations for SERBot.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.40):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path) if os.path.exists(model_path) else None
        
        # Classes considered potential physical obstacles on an industrial site
        self.obstacle_classes = {
            "person", "car", "motorcycle", "bus", "truck", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "backpack",
            "umbrella", "handbag", "suitcase", "bottle", "chair", "couch"
        }

    def estimate_distance_and_bearing(self, bbox: List[float], img_shape: Tuple[int, int], cls_name: str) -> Tuple[float, float, str]:
        """
        Estimates the distance in meters and angular bearing using pinhole camera geometry.
        
        Args:
            bbox: [x1, y1, x2, y2]
            img_shape: (height, width)
            cls_name: Object class name
            
        Returns:
            Tuple of (distance_meters, angle_degrees, bearing_str)
        """
        h_img, w_img = img_shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_height_px = max(1.0, y2 - y1)
        bbox_center_x = (x1 + x2) / 2.0
        
        # Real-world expected height of this object type
        real_h = REFERENCE_HEIGHTS.get(cls_name.lower(), REFERENCE_HEIGHTS["default"])
        focal_length = CAMERA_PARAMS["focal_length_px"]
        
        # Pinhole distance formula: d = (f * H_real) / h_image
        dist_m = (focal_length * real_h) / bbox_height_px
        
        # Ground plane perspective correction based on vertical image position
        # Objects touching the lower frame boundary are closer to the ground-level robot
        ground_factor = (y2 / h_img)
        if ground_factor > 0.85:
            dist_m = min(dist_m, (1.0 - ground_factor + 0.15) * 5.0)
            
        dist_m = max(0.3, round(float(dist_m), 2))
        
        # Calculate horizontal angle / bearing (-angle = left, +angle = right)
        img_center_x = w_img / 2.0
        offset_x = bbox_center_x - img_center_x
        # Field of view estimation (~65 deg horizontal)
        angle_deg = round((offset_x / img_center_x) * 32.5, 1)
        
        if angle_deg < -10.0:
            bearing = "LEFT"
        elif angle_deg > 10.0:
            bearing = "RIGHT"
        else:
            bearing = "CENTER"
            
        return dist_m, angle_deg, bearing

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects obstacles, computes spatial threat levels and navigation instructions.
        
        Returns:
            Dict containing detected obstacles, separate persons list, collision alerts,
            and SERBot navigation command.
        """
        if self.model is None:
            return {"obstacles": [], "persons": [], "alerts": [], "nav_command": "CLEAR"}
            
        h_img, w_img = frame.shape[:2]
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        obstacles = []
        persons = []
        alerts = []
        closest_distance = 999.0
        closest_obstacle = None
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            coords = [float(v) for v in box.xyxy[0]]
            
            # Distance and bearing
            dist_m, angle_deg, bearing = self.estimate_distance_and_bearing(coords, (h_img, w_img), cls_name)
            
            # Ground contact point (midpoint of bottom edge)
            ground_pt = [float((coords[0] + coords[2]) / 2.0), float(coords[3])]
            
            # Determine proximity danger level
            if dist_m <= PROXIMITY_RANGES["critical"]:
                proximity_level = "CRITICAL"
            elif dist_m <= PROXIMITY_RANGES["warning"]:
                proximity_level = "WARNING"
            elif dist_m <= PROXIMITY_RANGES["caution"]:
                proximity_level = "CAUTION"
            else:
                proximity_level = "SAFE"
                
            item = {
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": coords,
                "ground_point": ground_pt,
                "distance_m": dist_m,
                "bearing_angle_deg": angle_deg,
                "bearing": bearing,
                "proximity_level": proximity_level
            }
            
            if cls_name == "person":
                persons.append(item)
                
            if cls_name in self.obstacle_classes or dist_m <= PROXIMITY_RANGES["warning"]:
                obstacles.append(item)
                
                if dist_m < closest_distance:
                    closest_distance = dist_m
                    closest_obstacle = item
                    
                if proximity_level in ["CRITICAL", "WARNING"]:
                    alerts.append({
                        "type": "OBSTACLE_COLLISION_RISK",
                        "severity": proximity_level,
                        "obstacle": cls_name,
                        "distance_m": dist_m,
                        "bearing": bearing,
                        "message": f"Obstacle Alert: {cls_name.upper()} at {dist_m}m ({bearing})!",
                        "bbox": coords
                    })
        
        # Determine SERBot Navigation Command
        if closest_obstacle and closest_distance <= PROXIMITY_RANGES["critical"]:
            nav_command = "EMERGENCY_STOP"
            nav_status = f"HALT: {closest_obstacle['class']} directly ahead at {closest_distance}m"
        elif closest_obstacle and closest_distance <= PROXIMITY_RANGES["warning"]:
            if closest_obstacle["bearing"] == "CENTER":
                nav_command = "STEER_RIGHT" if closest_obstacle["bearing_angle_deg"] < 0 else "STEER_LEFT"
            elif closest_obstacle["bearing"] == "LEFT":
                nav_command = "STEER_RIGHT"
            else:
                nav_command = "STEER_LEFT"
            nav_status = f"SLOW & BYPASS: {closest_obstacle['class']} at {closest_distance}m"
        else:
            nav_command = "PROCEED_SAFE"
            nav_status = "Path clear for autonomous movement"
            
        return {
            "obstacles": obstacles,
            "persons": persons,
            "alerts": alerts,
            "closest_distance_m": closest_distance if closest_obstacle else None,
            "nav_command": nav_command,
            "nav_status": nav_status
        }
