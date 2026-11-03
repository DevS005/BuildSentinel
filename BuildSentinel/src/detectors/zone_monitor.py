"""
Unsafe Zones & Dynamic Geofencing Monitor
Monitors restricted zones, excavation perimeters, and crane drop radiuses for intrusions.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ..config import DEFAULT_UNSAFE_ZONES

class UnsafeZoneMonitor:
    """
    Geofence safety manager for hazardous construction and industrial areas.
    Detects workers, vehicles, or unauthorized objects breaching virtual perimeters.
    """
    def __init__(self, initial_zones: List[Dict[str, Any]] = None):
        self.zones = initial_zones if initial_zones is not None else DEFAULT_UNSAFE_ZONES

    def add_zone(self, zone_id: str, name: str, hazard_type: str, polygon_norm: List[List[float]], color: Tuple[int, int, int] = (0, 0, 255)):
        """Adds or updates an unsafe zone using normalized [0.0 - 1.0] polygon vertices."""
        self.zones.append({
            "id": zone_id,
            "name": name,
            "type": hazard_type,
            "color": color,
            "polygon": polygon_norm
        })

    def get_absolute_polygon(self, norm_poly: List[List[float]], img_shape: Tuple[int, int]) -> np.ndarray:
        """Converts normalized [0-1] polygon coordinates to pixel coordinates."""
        h, w = img_shape[:2]
        pts = [[int(pt[0] * w), int(pt[1] * h)] for pt in norm_poly]
        return np.array(pts, dtype=np.int32)

    def is_point_inside(self, point: Tuple[float, float], polygon_pts: np.ndarray) -> Tuple[bool, float]:
        """
        Tests if a point is inside a polygon using OpenCV pointPolygonTest.
        
        Returns:
            (is_inside, signed_distance)
            signed_distance > 0: inside, = 0: on edge, < 0: outside
        """
        dist = cv2.pointPolygonTest(polygon_pts, (float(point[0]), float(point[1])), measureDist=True)
        return (dist >= 0), float(dist)

    def monitor(self, frame: np.ndarray, tracked_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Checks all tracked entities (persons, obstacles, vehicles) against active unsafe zones.
        
        Args:
            frame: Input video frame
            tracked_entities: List of entities with 'ground_point' or 'bbox' and 'class'
            
        Returns:
            Dict containing active zone states, breach breaches, and intrusion warnings.
        """
        h_img, w_img = frame.shape[:2]
        breaches = []
        active_zone_status = []
        
        for zone in self.zones:
            abs_poly = self.get_absolute_polygon(zone["polygon"], (h_img, w_img))
            zone_intruders = []
            
            for entity in tracked_entities:
                # Use ground contact point (feet) if available, otherwise bottom center of bbox
                if "ground_point" in entity:
                    pt = entity["ground_point"]
                elif "bbox" in entity:
                    box = entity["bbox"]
                    pt = [(box[0] + box[2]) / 2.0, box[3]]
                else:
                    continue
                    
                is_inside, dist_px = self.is_point_inside(pt, abs_poly)
                
                if is_inside:
                    zone_intruders.append({
                        "entity_class": entity.get("class", "person"),
                        "ground_point": pt,
                        "penetration_depth_px": round(dist_px, 1),
                        "entity_info": entity
                    })
                    
                    breaches.append({
                        "type": "ZONE_INTRUSION",
                        "severity": "CRITICAL",
                        "zone_id": zone["id"],
                        "zone_name": zone["name"],
                        "hazard_type": zone["type"],
                        "intruder": entity.get("class", "person"),
                        "position": pt,
                        "message": f"DANGER: {entity.get('class', 'Person').capitalize()} entered {zone['name']}!"
                    })
            
            active_zone_status.append({
                "id": zone["id"],
                "name": zone["name"],
                "type": zone["type"],
                "polygon_abs": abs_poly.tolist(),
                "polygon_norm": zone["polygon"],
                "color": zone.get("color", (0, 0, 255)),
                "is_breached": len(zone_intruders) > 0,
                "intruder_count": len(zone_intruders)
            })
            
        return {
            "zones": active_zone_status,
            "breaches": breaches,
            "total_breaches": len(breaches),
            "status": "SECURE" if len(breaches) == 0 else "ZONE_VIOLATION"
        }
