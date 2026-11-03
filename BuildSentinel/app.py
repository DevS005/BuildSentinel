"""
SERBot BuildSentinel - Web Dashboard & Live Robotic Perception Server
Flask server providing REST APIs, live video stream, and an interactive safety portal.
"""
from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import os
import time
from pathlib import Path
from src.pipeline import SERBotPerceptionEngine

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

# Initialize Master Perception Engine
print("[Server] Initializing SERBot Perception Engine...")
engine = SERBotPerceptionEngine()
print("[Server] Engine Ready.")

DEMO_DIR = Path("demo_assets")

def frame_to_base64(frame: np.ndarray) -> str:
    """Encodes BGR numpy image to JPEG base64 data URI."""
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ret:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "ONLINE",
        "engine": "SERBot BuildSentinel v1.0",
        "models_loaded": {
            "ppe": engine.ppe_detector.model_ppe is not None,
            "ppe_vest": engine.ppe_detector.model_vest is not None,
            "obstacle": engine.obstacle_detector.model is not None,
            "fire_smoke": engine.hazard_detector.model_fire is not None,
            "crack": engine.structural_monitor.model_crack is not None
        }
    })

@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Lists available demo site imagery."""
    samples = []
    if DEMO_DIR.exists():
        for file in DEMO_DIR.glob("*.jpg"):
            if not file.name.startswith("annotated_"):
                samples.append({
                    "id": file.name,
                    "name": file.stem.replace("_", " ").title(),
                    "filename": file.name
                })
    return jsonify({"samples": samples})

@app.route("/api/analyze", methods=["POST"])
def analyze_frame():
    """
    Analyzes an uploaded image or pre-loaded sample.
    """
    img = None
    
    # 1. Check if user sent a file upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != '':
            in_memory = file.read()
            nparr = np.frombuffer(in_memory, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
    # 2. Check if user requested a demo sample
    elif request.is_json:
        data = request.get_json()
        sample_id = data.get("sample_id")
        if sample_id:
            sample_path = DEMO_DIR / sample_id
            if sample_path.exists():
                img = cv2.imread(str(sample_path))

    if img is None:
        return jsonify({"error": "No valid image provided or file unreadable."}), 400

    # Execute end-to-end perception pipeline
    report = engine.process_frame(img, render_hud=True)
    
    # Encode annotated frame as base64
    annotated_b64 = frame_to_base64(report["annotated_frame"])
    
    # Prepare serializable response
    clean_report = {k: v for k, v in report.items() if k != "annotated_frame"}
    clean_report["annotated_image_b64"] = annotated_b64
    clean_report["image_width"] = img.shape[1]
    clean_report["image_height"] = img.shape[0]

    return jsonify(clean_report)

@app.route("/api/zones", methods=["GET", "POST"])
def handle_zones():
    """Manage dynamic geofence unsafe zones."""
    if request.method == "POST":
        data = request.get_json()
        if not data or "polygon" not in data:
            return jsonify({"error": "Invalid zone payload"}), 400
            
        zone_id = data.get("id", f"zone_{int(time.time())}")
        name = data.get("name", "Custom Hazard Area")
        hazard_type = data.get("type", "restricted_entry")
        poly = data.get("polygon")
        
        engine.zone_monitor.add_zone(zone_id, name, hazard_type, poly)
        return jsonify({"success": True, "message": f"Zone '{name}' added successfully!"})
        
    return jsonify({"zones": engine.zone_monitor.zones})

def generate_video_stream():
    """Streams live webcam feed or looped simulation with real-time HUD."""
    # Attempt webcam, fallback to synthetic generator
    cap = cv2.VideoCapture(0)
    has_cam = cap.isOpened()
    
    # Fallback image loop
    sample_files = list(DEMO_DIR.glob("*.jpg"))
    sample_files = [f for f in sample_files if not f.name.startswith("annotated_")]
    idx = 0
    
    try:
        while True:
            if has_cam:
                ret, frame = cap.read()
                if not ret:
                    has_cam = False
                    continue
            else:
                # Cycle through demo sample images with simulated sensor dynamics
                if sample_files:
                    path = sample_files[idx % len(sample_files)]
                    frame = cv2.imread(str(path))
                    # Add tiny motion noise to keep optical flow active
                    frame = np.roll(frame, shift=int(np.sin(time.time() * 2) * 3), axis=1)
                    idx += 1
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                time.sleep(0.4)
                
            report = engine.process_frame(frame, render_hud=True)
            annotated = report["annotated_frame"]
            
            ret, jpeg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ret:
                continue
                
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if cap.isOpened():
            cap.release()

@app.route("/api/video_feed")
def video_feed():
    """MJPEG stream endpoint for live monitoring."""
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("[*] Starting SERBot BuildSentinel Web Dashboard on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
