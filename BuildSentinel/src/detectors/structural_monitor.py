"""
Structural Health & Movement Monitoring Module
Detects surface cracks, spalling, structural displacement, tilts, and vibration shifts.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import os
from ..config import STRUCTURAL_TRACKING

class StructuralMonitor:
    """
    Dual-engine structural health monitor:
    1. AI Deep Learning: Detects structural cracks, fissures, and degradation.
    2. Optical Flow & Feature Displacement: Measures micromovements, shifts, tilts, and vibrations.
    """
    def __init__(self, crack_model_path: str, conf_threshold: float = 0.30):
        self.conf_threshold = conf_threshold
        self.model_crack = YOLO(crack_model_path) if os.path.exists(crack_model_path) else None
        
        # Tracking state for structural movement
        self.prev_gray = None
        self.prev_points = None
        self.cumulative_displacement = 0.0
        self.history_displacements = []
        self.movement_threshold = STRUCTURAL_TRACKING["displacement_alert_threshold_px"]

    def detect_cracks(self, frame: np.ndarray) -> Dict[str, Any]:
        """Runs crack detection model and estimates crack severity."""
        cracks = []
        alerts = []
        
        if self.model_crack is not None:
            results = self.model_crack(frame, conf=self.conf_threshold, verbose=False)[0]
            for box in results.boxes:
                conf = float(box.conf[0])
                coords = [float(v) for v in box.xyxy[0]]
                w = coords[2] - coords[0]
                h = coords[3] - coords[1]
                length = max(w, h)
                
                # Classify severity based on crack size & bounding dimension
                if length > 120:
                    severity = "CRITICAL"
                elif length > 50:
                    severity = "MODERATE"
                else:
                    severity = "MINOR"
                    
                crack_info = {
                    "class": "structural_crack",
                    "confidence": round(conf, 3),
                    "bbox": coords,
                    "estimated_length_px": round(length, 1),
                    "severity": severity
                }
                cracks.append(crack_info)
                
                if severity in ["CRITICAL", "MODERATE"]:
                    alerts.append({
                        "type": "STRUCTURAL_DEFECT",
                        "severity": severity,
                        "defect": "Surface Crack",
                        "estimated_length_px": round(length, 1),
                        "message": f"Structural Alert: {severity} Crack detected (est. {round(length, 1)}px length)!",
                        "bbox": coords
                    })
                    
        return {
            "cracks": cracks,
            "total_cracks": len(cracks),
            "alerts": alerts
        }

    def track_structural_movement(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Calculates frame-to-frame structural displacement using Lucas-Kanade Optical Flow.
        Flags unusual structural shifts, tilts, or continuous deflections.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shift_vector = [0.0, 0.0]
        displacement_mag = 0.0
        tracked_vectors = []
        movement_alert = None
        
        # If first frame, shape changed, or tracked points lost, detect good features to track
        if (self.prev_gray is None or self.prev_points is None or 
            len(self.prev_points) < 15 or self.prev_gray.shape != gray.shape):
            # Detect Shi-Tomasi corners on structural features
            points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=STRUCTURAL_TRACKING["max_corners"],
                qualityLevel=STRUCTURAL_TRACKING["quality_level"],
                minDistance=STRUCTURAL_TRACKING["min_distance"],
                blockSize=STRUCTURAL_TRACKING["block_size"]
            )
            self.prev_gray = gray
            self.prev_points = points
            return {
                "movement_detected": False,
                "displacement_px": 0.0,
                "vector": [0.0, 0.0],
                "flow_vectors": [],
                "alert": None
            }
            
        # Calculate optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        if curr_points is not None and status is not None:
            # Select good points
            good_curr = curr_points[status == 1]
            good_prev = self.prev_points[status == 1]
            
            if len(good_curr) > 0:
                # Calculate displacement for each tracked feature
                diffs = good_curr - good_prev
                dx_mean = float(np.mean(diffs[:, 0]))
                dy_mean = float(np.mean(diffs[:, 1]))
                displacement_mag = float(np.sqrt(dx_mean**2 + dy_mean**2))
                shift_vector = [round(dx_mean, 2), round(dy_mean, 2)]
                
                # Keep up to 30 flow vectors for visualization HUD
                step = max(1, len(good_curr) // 30)
                for i in range(0, len(good_curr), step):
                    p0 = [float(good_prev[i][0]), float(good_prev[i][1])]
                    p1 = [float(good_curr[i][0]), float(good_curr[i][1])]
                    tracked_vectors.append({"start": p0, "end": p1})
                    
                # Track cumulative history for oscillation/vibration detection
                self.history_displacements.append(displacement_mag)
                if len(self.history_displacements) > 30:
                    self.history_displacements.pop(0)
                    
                if displacement_mag >= self.movement_threshold:
                    movement_alert = {
                        "type": "STRUCTURAL_SHIFT",
                        "severity": "CRITICAL" if displacement_mag > (self.movement_threshold * 2) else "WARNING",
                        "displacement_px": round(displacement_mag, 2),
                        "vector": shift_vector,
                        "message": f"STRUCTURAL ALERT: Movement of {round(displacement_mag, 2)}px detected ({shift_vector})!"
                    }
                    
                # Update points for next cycle
                self.prev_points = good_curr.reshape(-1, 1, 2)
            else:
                self.prev_points = None
        else:
            self.prev_points = None
            
        self.prev_gray = gray
        
        return {
            "movement_detected": displacement_mag >= 1.0,
            "displacement_px": round(displacement_mag, 2),
            "vector": shift_vector,
            "flow_vectors": tracked_vectors,
            "alert": movement_alert
        }

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Unified structural assessment combining cracks and movement."""
        crack_res = self.detect_cracks(frame)
        move_res = self.track_structural_movement(frame)
        
        all_alerts = crack_res["alerts"].copy()
        if move_res["alert"]:
            all_alerts.append(move_res["alert"])
            
        return {
            "cracks": crack_res["cracks"],
            "total_cracks": crack_res["total_cracks"],
            "movement": move_res,
            "alerts": all_alerts,
            "structural_status": "NORMAL" if len(all_alerts) == 0 else "DEFECT_DETECTED"
        }
