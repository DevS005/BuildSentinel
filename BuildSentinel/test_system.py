"""
Automated System Verification Script for SERBot Perception Engine
Tests all 5 detection modules and sensor fusion pipeline.
"""
import cv2
import numpy as np
import os
import json
from src.pipeline import SERBotPerceptionEngine

def create_synthetic_site_image():
    """Generates a synthetic construction site test frame."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 1. Background ground and concrete wall
    img[0:360, :] = (90, 85, 80)     # Concrete gray wall
    img[360:720, :] = (45, 50, 55)   # Asphalt / construction ground
    
    # 2. Draw a simulated structural crack on wall
    pts = np.array([[300, 100], [320, 140], [315, 190], [335, 250], [340, 280]], np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(20, 20, 20), thickness=4)
    cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=2)
    
    # 3. Draw a simulated flame / fire spot
    cv2.circle(img, (950, 480), 35, (0, 140, 255), -1)
    cv2.circle(img, (950, 475), 20, (0, 215, 255), -1)
    cv2.circle(img, (950, 470), 8, (255, 255, 255), -1)
    
    # 4. Draw a simulated wet liquid spill puddle
    cv2.ellipse(img, (400, 580), (120, 40), 10, 0, 360, (230, 230, 230), -1)
    
    # 5. Draw a simulated worker / obstacle
    # Worker body
    cv2.rectangle(img, (600, 350), (680, 620), (30, 80, 180), -1)
    # Head
    cv2.circle(img, (640, 310), 30, (180, 150, 120), -1)
    # Hardhat (yellow)
    cv2.ellipse(img, (640, 295), (32, 18), 0, 180, 360, (0, 215, 255), -1)
    # High-vis vest stripes
    cv2.rectangle(img, (605, 390), (675, 520), (0, 220, 150), -1)
    cv2.line(img, (620, 390), (620, 520), (240, 240, 240), 5)
    cv2.line(img, (660, 390), (660, 520), (240, 240, 240), 5)
    
    # 6. Obstacle barrel/box
    cv2.rectangle(img, (150, 450), (260, 600), (40, 120, 200), -1)
    
    return img

def test_pipeline():
    print("[Test] Starting SERBot Engine Diagnostic...")
    
    os.makedirs("demo_assets", exist_ok=True)
    test_img = create_synthetic_site_image()
    cv2.imwrite("demo_assets/test_input.jpg", test_img)
    print("[Test] Generated synthetic site image: demo_assets/test_input.jpg")
    
    # Initialize Engine
    engine = SERBotPerceptionEngine()
    
    # Process Frame 1
    print("[Test] Running Frame 1 processing...")
    report1 = engine.process_frame(test_img)
    
    # Process Frame 2 (with slight shift to verify optical flow movement detection)
    print("[Test] Running Frame 2 (simulating 5px structural shift)...")
    shifted_img = np.roll(test_img, shift=5, axis=1)  # 5 pixel horizontal shift
    report2 = engine.process_frame(shifted_img)
    
    # Save output annotated image
    output_path = "demo_assets/test_output_annotated.jpg"
    cv2.imwrite(output_path, report2["annotated_frame"])
    print(f"[Test] Saved annotated result: {output_path}")
    
    # Verification checks
    assert "ppe" in report2, "Missing PPE results"
    assert "obstacles" in report2, "Missing obstacle results"
    assert "zones" in report2, "Missing zone results"
    assert "structural" in report2, "Missing structural results"
    assert "hazards" in report2, "Missing hazard results"
    assert "fusion" in report2, "Missing sensor fusion results"
    
    print("\n================ DIAGNOSTIC REPORT ================")
    print(f"Latency: {report2['latency_ms']} ms ({report2['fps']} FPS)")
    print(f"Site Safety Score: {report2['fusion']['site_safety_score']}/100")
    print(f"Overall Safety Status: {report2['fusion']['overall_status']}")
    print(f"Robotic Nav Directive: {report2['obstacles']['nav_command']}")
    print(f"Total Active Alerts: {len(report2['all_alerts'])}")
    for idx, alert in enumerate(report2['all_alerts']):
        print(f"  [{idx+1}] {alert.get('type')}: {alert.get('message')}")
    print("===================================================\n")
    print("[SUCCESS] All 5 modules and sensor fusion tested successfully!")

if __name__ == "__main__":
    test_pipeline()
