"""
HUD & Multi-Hazard Visualizer
Renders futuristic augmented telemetry overlays, bounding boxes, geofences,
optical flow vectors, and compliance badges onto image frames.
"""
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .config import COLOR_PALETTE

class HUDVisualizer:
    """
    Renders high-contrast, informative robotic HUD overlays
    for real-time display, recording, and dashboard streaming.
    """
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, perception_output: Dict[str, Any], show_zones: bool = True, show_hud: bool = True) -> np.ndarray:
        """
        Draws all detection results onto a copy of the frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Draw Unsafe Geofence Zones (Semi-transparent overlay)
        if show_zones and "zones" in perception_output:
            overlay = canvas.copy()
            for zone in perception_output["zones"]["zones"]:
                poly = np.array(zone["polygon_abs"], dtype=np.int32)
                is_breached = zone["is_breached"]
                
                # Fill color: Red if breached, Amber/Orange if idle
                fill_color = (0, 0, 220) if is_breached else (0, 140, 255)
                cv2.fillPoly(overlay, [poly], fill_color)
                
                # Boundary outline
                line_color = (0, 0, 255) if is_breached else (0, 200, 255)
                cv2.polylines(canvas, [poly], isClosed=True, color=line_color, thickness=2)
                
                # Label
                centroid_x = int(np.mean(poly[:, 0]))
                centroid_y = int(np.mean(poly[:, 1]))
                status_txt = f"! BREACH: {zone['name']} !" if is_breached else zone["name"]
                cv2.putText(canvas, status_txt, (max(10, centroid_x - 80), centroid_y),
                            self.font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                            
            # Blend polygon overlay
            alpha = 0.25
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # 2. Draw Optical Flow Movement Vectors (Structural tracking)
        structural_data = perception_output.get("structural", {})
        flow_vectors = structural_data.get("movement", {}).get("flow_vectors", [])
        for vec in flow_vectors:
            pt1 = (int(vec["start"][0]), int(vec["start"][1]))
            pt2 = (int(vec["end"][0]), int(vec["end"][1]))
            cv2.arrowedLine(canvas, pt1, pt2, (255, 230, 0), 1, tipLength=0.3)

        # 3. Draw Structural Cracks
        cracks = structural_data.get("cracks", [])
        for c in cracks:
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            sev = c["severity"]
            box_col = (0, 0, 255) if sev == "CRITICAL" else (0, 165, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_col, 2)
            lbl = f"CRACK [{sev} | L:{c['estimated_length_px']}px]"
            self._draw_badge(canvas, lbl, (x1, y1 - 8), box_col)

        # 4. Draw Environmental Hazards (Fire, Smoke, Spills)
        hazards = perception_output.get("hazards", {}).get("hazards", [])
        for haz in hazards:
            x1, y1, x2, y2 = [int(v) for v in haz["bbox"]]
            is_fire = "fire" in haz["class"].lower() or "flame" in haz["class"].lower()
            h_col = (0, 69, 255) if is_fire else (0, 200, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), h_col, 2)
            lbl = f"HAZARD: {haz['class'].upper()}"
            self._draw_badge(canvas, lbl, (x1, y1 - 8), h_col)

        # 5. Draw Obstacles
        obstacles = perception_output.get("obstacles", {}).get("obstacles", [])
        for obs in obstacles:
            if obs["class"] == "person":
                continue  # Handled in PPE section for enriched worker data
            x1, y1, x2, y2 = [int(v) for v in obs["bbox"]]
            prox = obs["proximity_level"]
            o_col = (0, 0, 255) if prox == "CRITICAL" else ((0, 165, 255) if prox == "WARNING" else (200, 200, 200))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), o_col, 2)
            lbl = f"{obs['class'].upper()} [{obs['distance_m']}m | {obs['bearing']}]"
            self._draw_badge(canvas, lbl, (x1, y1 - 8), o_col)

        # 6. Draw Worker PPE Compliance
        worker_reports = perception_output.get("ppe", {}).get("worker_reports", [])
        for w_rep in worker_reports:
            x1, y1, x2, y2 = [int(v) for v in w_rep["bbox"]]
            is_comp = w_rep["is_compliant"]
            w_col = COLOR_PALETTE["safe"] if is_comp else COLOR_PALETTE["critical"]
            
            # Draw corner brackets for worker
            self._draw_corner_rect(canvas, (x1, y1, x2, y2), w_col, thickness=2, length=18)
            
            # Worker Compliance Tag
            if is_comp:
                lbl = f"WORKER #{w_rep['worker_id']} [COMPLIANT]"
            else:
                missing_str = "+".join(w_rep["missing_gear"]).upper()
                lbl = f"WORKER #{w_rep['worker_id']} [VIOLATION: NO {missing_str}]"
            self._draw_badge(canvas, lbl, (x1, y1 - 10), w_col)

        # 7. Draw Raw PPE items (Helmets, Vests, Gloves, etc.)
        raw_ppe = perception_output.get("ppe", {}).get("raw_detections", [])
        for item in raw_ppe:
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            cls = item["class"].lower()
            is_violation = cls.startswith("no_") or cls.startswith("no-")
            col = (0, 0, 255) if is_violation else (0, 230, 115)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1)
            cv2.putText(canvas, item["class"], (x1, max(12, y1 - 4)), self.font, 0.4, col, 1, cv2.LINE_AA)

        # 8. Render Futuristic Top & Bottom HUD Banners
        if show_hud:
            self._render_hud_header_and_footer(canvas, perception_output)

        return canvas

    def _draw_badge(self, img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """Draws filled label badge with crisp dark text."""
        x, y = pos
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, self.font, font_scale, thickness)
        top_left = (x, max(0, y - th - 4))
        bottom_right = (x + tw + 8, max(th + 4, y + 2))
        cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(img, text, (x + 4, bottom_right[1] - 4), self.font, font_scale, (10, 10, 15), thickness, cv2.LINE_AA)

    def _draw_corner_rect(self, img: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 2, length: int = 15):
        """Draws high-tech corner brackets around an object."""
        x1, y1, x2, y2 = bbox
        # Top-left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def _render_hud_header_and_footer(self, img: np.ndarray, data: Dict[str, Any]):
        """Draws top system bar and bottom telemetry gauges."""
        h, w = img.shape[:2]
        
        # --- Top HUD Header Bar ---
        top_bar = np.zeros((40, w, 3), dtype=np.uint8)
        top_bar[:] = (18, 20, 24)
        cv2.addWeighted(img[0:40, 0:w], 0.3, top_bar, 0.7, 0, img[0:40, 0:w])
        
        # System Title
        cv2.putText(img, "SERBot AI SIGHT v1.0", (15, 26), self.font, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        
        # Safety Status Indicator
        status = data.get("fusion", {}).get("overall_status", "ACTIVE")
        score = data.get("fusion", {}).get("site_safety_score", 100)
        
        status_color = (0, 230, 115) if "OPTIMAL" in status else ((0, 165, 255) if "ELEVATED" in status else (0, 0, 255))
        status_txt = f"STATUS: {status} | SCORE: {score}/100"
        cv2.putText(img, status_txt, (w - 380, 26), self.font, 0.5, status_color, 2, cv2.LINE_AA)
        
        # Navigation Directive
        nav_cmd = data.get("obstacles", {}).get("nav_command", "CLEAR")
        nav_color = (0, 0, 255) if "STOP" in nav_cmd else ((0, 200, 255) if "STEER" in nav_cmd else (0, 230, 115))
        cv2.putText(img, f"NAV: {nav_cmd}", (w // 2 - 80, 26), self.font, 0.5, nav_color, 2, cv2.LINE_AA)
        
        # --- Bottom HUD Footer Bar ---
        bot_h = 32
        bot_bar = np.zeros((bot_h, w, 3), dtype=np.uint8)
        bot_bar[:] = (18, 20, 24)
        cv2.addWeighted(img[h - bot_h:h, 0:w], 0.3, bot_bar, 0.7, 0, img[h - bot_h:h, 0:w])
        
        # Telemetry string
        telemetry = data.get("fusion", {}).get("telemetry", {})
        telem_str = (
            f"GAS: {telemetry.get('gas_smoke_ppm', '--')} ppm  |  "
            f"TEMP: {telemetry.get('temperature_c', '--')}C  |  "
            f"IMU: {telemetry.get('vibration_g', '--')}g  |  "
            f"PROX: {telemetry.get('ultrasonic_distance_cm', '--')} cm  |  "
            f"BAT: {telemetry.get('battery_pct', '--')}%"
        )
        cv2.putText(img, telem_str, (15, h - 10), self.font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
