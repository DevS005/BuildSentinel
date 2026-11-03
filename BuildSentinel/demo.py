"""
SERBot BuildSentinel CLI Demonstration Runner
Run multi-hazard site safety perception on images, videos, or live webcams.

Usage:
    python demo.py --image demo_assets/construction_workers.jpg
    python demo.py --image demo_assets/heavy_machinery.jpg --save output.jpg
    python demo.py --webcam
    python demo.py --video path/to/site_walkthrough.mp4
"""
import argparse
import cv2
import json
import sys
from pathlib import Path
from src.pipeline import SERBotPerceptionEngine

def main():
    parser = argparse.ArgumentParser(description="SERBot AI Vision & Safety Perception Engine")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Run on live camera stream (Device 0)")
    parser.add_argument("--save", type=str, default=None, help="Path to save annotated output")
    parser.add_argument("--json", action="store_true", help="Print structured JSON telemetry to stdout")
    parser.add_argument("--no-hud", action="store_true", help="Disable HUD banner overlay")
    args = parser.parse_args()

    # Default to sample image if no argument provided
    if not args.image and not args.video and not args.webcam:
        default_sample = "demo_assets/construction_workers.jpg"
        print(f"[*] No input specified. Defaulting to sample: {default_sample}")
        args.image = default_sample

    engine = SERBotPerceptionEngine()

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[Error] Image not found: {img_path}")
            sys.exit(1)
            
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[Error] Failed to decode image: {img_path}")
            sys.exit(1)

        print(f"[*] Processing {img_path.name} ({frame.shape[1]}x{frame.shape[0]})...")
        report = engine.process_frame(frame, render_hud=not args.no_hud)

        print("\n" + "="*50)
        print("          SERBot SITE SAFETY REPORT          ")
        print("="*50)
        print(f"Processing Time   : {report['latency_ms']} ms ({report['fps']} FPS)")
        print(f"Site Safety Index : {report['fusion']['site_safety_score']} / 100 ({report['fusion']['overall_status']})")
        print(f"Nav Directive     : {report['obstacles']['nav_command']} - {report['obstacles']['nav_status']}")
        print(f"Workers Detected  : {len(report['ppe']['worker_reports'])}")
        print(f"PPE Compliance    : {report['ppe']['compliance_rate']}%")
        print(f"Obstacles Count   : {len(report['obstacles']['obstacles'])}")
        print(f"Cracks / Defects  : {len(report['structural']['cracks'])}")
        print(f"Env. Hazards      : {len(report['hazards']['hazards'])}")
        print(f"Geofence Breaches : {report['zones']['total_breaches']}")
        print(f"Active Alerts     : {len(report['all_alerts'])}")
        print("="*50)
        
        for idx, alert in enumerate(report["all_alerts"]):
            print(f"  [{idx+1}] [{alert.get('severity', 'INFO')}] {alert.get('message')}")
        print("="*50 + "\n")

        if args.json:
            # Serialize report without numpy frame
            json_report = {k: v for k, v in report.items() if k != "annotated_frame"}
            print(json.dumps(json_report, default=str, indent=2))

        # Save output
        out_dest = args.save if args.save else f"demo_assets/annotated_{img_path.name}"
        cv2.imwrite(out_dest, report["annotated_frame"])
        print(f"[+] Annotated visual saved to: {out_dest}")

    elif args.webcam or args.video:
        src = 0 if args.webcam else args.video
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[Error] Could not open video source: {src}")
            sys.exit(1)

        print(f"[*] Streaming from {src}. Press 'q' to stop.")
        writer = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                report = engine.process_frame(frame, render_hud=not args.no_hud)
                display_frame = report["annotated_frame"]

                if args.save:
                    if writer is None:
                        h, w = display_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w, h))
                    writer.write(display_frame)

                cv2.imshow("SERBot AI Perception - Site Safety & Structural Monitor", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("[*] Stream terminated.")

if __name__ == "__main__":
    main()
