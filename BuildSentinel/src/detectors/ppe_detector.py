"""
PPE & Safety Gear Compliance Detector
Detects safety gear presence and flags safety gaps (missing helmets, vests, gloves, goggles, shoes).
"""
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import os

class PPEDetector:
    """
    Evaluates worker safety compliance using deep vision models.
    Supports dual-model ensemble for hardhat/no-hardhat, safety vests, gloves, and shoes.
    """
    def __init__(self, ppe_model_path: str, vest_model_path: str = None, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        
        # Primary PPE model (detects helmet, no_helmet, glove, no_glove, goggles, shoes, etc.)
        self.model_ppe = YOLO(ppe_model_path) if os.path.exists(ppe_model_path) else None
        
        # Secondary model (specialized in High-Vis Vest detection)
        self.model_vest = YOLO(vest_model_path) if vest_model_path and os.path.exists(vest_model_path) else None
        
        # Define required PPE checklist for standard construction/industrial site
        self.mandatory_ppe = ["helmet", "vest"]
        self.recommended_ppe = ["goggles", "gloves", "safety_shoes"]

    def _box_contains(self, parent_box: List[float], child_box: List[float], threshold_overlap: float = 0.35) -> bool:
        """Checks if child_box is contained within or overlaps with parent_box (e.g., worker)."""
        px1, py1, px2, py2 = parent_box
        cx1, cy1, cx2, cy2 = child_box
        
        # Child center point
        ccx = (cx1 + cx2) / 2.0
        ccy = (cy1 + cy2) / 2.0
        
        # Center containment test
        if px1 <= ccx <= px2 and py1 <= ccy <= py2:
            return True
            
        # Intersection over child area
        ix1 = max(px1, cx1)
        iy1 = max(py1, cy1)
        ix2 = min(px2, cx2)
        iy2 = min(py2, cy2)
        
        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            child_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
            if (intersection / child_area) >= threshold_overlap:
                return True
        return False

    def detect(self, frame: np.ndarray, person_boxes: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run PPE detection on input frame.
        
        Args:
            frame: BGR numpy image
            person_boxes: Optional list of pre-detected person bounding boxes
            
        Returns:
            Dict containing raw detections, worker compliance reports, and safety violations.
        """
        raw_items = []
        violations = []
        worker_reports = []
        
        # 1. Run Primary PPE Model
        if self.model_ppe is not None:
            results_ppe = self.model_ppe(frame, conf=self.conf_threshold, verbose=False)[0]
            for box in results_ppe.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model_ppe.names[cls_id]
                conf = float(box.conf[0])
                coords = [float(v) for v in box.xyxy[0]]
                raw_items.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": coords,
                    "source": "ppe_primary"
                })
        
        # 2. Run Secondary Vest Model if available
        if self.model_vest is not None:
            results_vest = self.model_vest(frame, conf=self.conf_threshold, verbose=False)[0]
            for box in results_vest.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model_vest.names[cls_id].lower()
                conf = float(box.conf[0])
                coords = [float(v) for v in box.xyxy[0]]
                if "vest" in cls_name:
                    raw_items.append({
                        "class": "vest",
                        "confidence": round(conf, 3),
                        "bbox": coords,
                        "source": "ppe_vest"
                    })
        
        # 3. Direct violation detection from negative classes ('no_helmet', 'no_glove', 'no_goggles', 'no_shoes')
        for item in raw_items:
            cls = item["class"].lower()
            if cls.startswith("no_") or cls.startswith("no-"):
                clean_gear = cls.replace("no_", "").replace("no-", "")
                violations.append({
                    "type": "MISSING_PPE",
                    "gear": clean_gear,
                    "severity": "CRITICAL" if clean_gear in ["helmet", "vest"] else "WARNING",
                    "message": f"Active Violation: Missing {clean_gear.capitalize()} detected!",
                    "bbox": item["bbox"],
                    "confidence": item["confidence"]
                })
        
        # 4. Associate detected PPE with individual workers (if person boxes are available)
        if person_boxes:
            for idx, person in enumerate(person_boxes):
                p_box = person["bbox"]
                w, h = p_box[2] - p_box[0], p_box[3] - p_box[1]
                
                # Partition body into head (top 25%), torso (20%-75%), and feet (70%-100%)
                head_zone = [p_box[0], p_box[1], p_box[2], p_box[1] + 0.30 * h]
                torso_zone = [p_box[0], p_box[1] + 0.20 * h, p_box[2], p_box[1] + 0.75 * h]
                feet_zone = [p_box[0], p_box[1] + 0.65 * h, p_box[2], p_box[3]]
                
                detected_gear = []
                missing_gear = []
                
                # Check for helmet in head zone
                has_helmet = any(
                    item["class"] == "helmet" and self._box_contains(head_zone, item["bbox"], 0.2)
                    for item in raw_items
                )
                has_no_helmet = any(
                    item["class"] in ["no_helmet", "no-helmet"] and self._box_contains(head_zone, item["bbox"], 0.2)
                    for item in raw_items
                )
                
                if has_helmet:
                    detected_gear.append("helmet")
                elif has_no_helmet or len(raw_items) > 0:
                    missing_gear.append("helmet")
                
                # Check for vest in torso zone
                has_vest = any(
                    "vest" in item["class"].lower() and self._box_contains(torso_zone, item["bbox"], 0.25)
                    for item in raw_items
                )
                if has_vest:
                    detected_gear.append("vest")
                else:
                    # If person is large enough (> 120px tall) and no vest found, flag missing vest
                    if h > 120:
                        missing_gear.append("vest")
                        
                # Check for shoes
                has_shoes = any(
                    item["class"] in ["shoes", "safety_shoe"] and self._box_contains(feet_zone, item["bbox"], 0.2)
                    for item in raw_items
                )
                if has_shoes:
                    detected_gear.append("safety_shoes")
                
                is_compliant = ("helmet" in detected_gear) and ("vest" in detected_gear) and (len(missing_gear) == 0)
                
                worker_report = {
                    "worker_id": idx + 1,
                    "bbox": p_box,
                    "confidence": person["confidence"],
                    "is_compliant": is_compliant,
                    "detected_gear": detected_gear,
                    "missing_gear": missing_gear,
                    "status": "COMPLIANT" if is_compliant else "NON_COMPLIANT"
                }
                worker_reports.append(worker_report)
                
                if not is_compliant and missing_gear:
                    violations.append({
                        "type": "WORKER_NON_COMPLIANT",
                        "worker_id": idx + 1,
                        "gear": ", ".join(missing_gear),
                        "severity": "CRITICAL" if any(g in ["helmet", "vest"] for g in missing_gear) else "WARNING",
                        "message": f"Worker #{idx + 1} Missing: {', '.join(missing_gear)}",
                        "bbox": p_box,
                        "confidence": person["confidence"]
                    })
        
        return {
            "raw_detections": raw_items,
            "worker_reports": worker_reports,
            "violations": violations,
            "total_violations": len(violations),
            "compliance_rate": self._compute_compliance_rate(worker_reports)
        }

    def _compute_compliance_rate(self, worker_reports: List[Dict[str, Any]]) -> float:
        if not worker_reports:
            return 100.0
        compliant_count = sum(1 for w in worker_reports if w["is_compliant"])
        return round((compliant_count / len(worker_reports)) * 100.0, 1)
