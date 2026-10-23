"""
AI-Powered Driver Monitoring & Safety Assistant
Main Execution Entry Point
"""

import os
import warnings

# Suppress Protobuf, TensorFlow, MediaPipe, and Intel OpenMP shutdown noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import argparse
import webbrowser
import threading
import time
import cv2
import uvicorn
import numpy as np


class SuppressCancelledFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and "CancelledError" in str(record.exc_info[0]):
            return False
        if "CancelledError" in record.getMessage():
            return False
        return True


logging.getLogger("uvicorn.error").addFilter(SuppressCancelledFilter())

from config import SERVER_HOST, SERVER_PORT
from ui.web_app import app, pipeline


def run_web_dashboard(auto_open: bool = True):
    url = f"http://localhost:{SERVER_PORT}"
    print("\n=======================================================")
    print("  GuardianDrive AI — Real-Time Driver Safety Co-Pilot  ")
    print(f"  Web Dashboard running at: {url}                      ")
    print("=======================================================\n")

    if auto_open:
        def _open():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n[GuardianDrive AI] Server stopped gracefully. Safe travels!")
        pipeline.stop()


def run_desktop_opencv():
    print("\n[GuardianDrive AI] Launching Standalone Desktop OpenCV Window...")
    print("Press 'q' or ESC in the video window to exit.")
    print("Press 'c' to calibrate full driver baseline (EAR, MAR, Head Pose & Emotion).\n")

    # Stop background web worker thread so desktop loop has exclusive access to detectors
    pipeline.stop_background_worker()

    fps_tracker = []
    last_time = time.time()

    try:
        while True:
            frame = pipeline.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            if dt > 0:
                fps_tracker.append(1.0 / dt)
                if len(fps_tracker) > 30:
                    fps_tracker.pop(0)
            avg_fps = float(np.mean(fps_tracker)) if fps_tracker else 30.0

            # Unified pipeline frame processing (landmarks, drowsiness, distraction, emotion, phone, risk, voice, DB)
            display_frame = pipeline._process_frame(frame, avg_fps, current_time)

            cv2.imshow("GuardianDrive AI - In-Cabin HUD", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                raw_ear = pipeline.latest_telemetry.get("ear", 0.28)
                raw_mar = pipeline.latest_telemetry.get("mar", 0.20)
                yaw = pipeline.latest_telemetry.get("yaw", 0.0)
                pitch = pipeline.latest_telemetry.get("pitch", 0.0)
                roll = pipeline.latest_telemetry.get("roll", 0.0)
                pipeline.drowsiness_detector.calibrate(baseline_ear=raw_ear, baseline_mar=raw_mar)
                pipeline.distraction_detector.calibrate(yaw, pitch, roll)
                if pipeline.latest_landmarks is not None:
                    pipeline.emotion_detector.calibrate(pipeline.latest_landmarks)
                print("[Info] Full driver baseline (EAR, MAR, Head Pose, Emotion) calibrated.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description="GuardianDrive AI Driver Safety System")
    parser.add_argument("--desktop", action="store_true", help="Launch OpenCV desktop window instead of web server")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open default browser on start")
    args = parser.parse_args()

    try:
        if args.desktop:
            run_desktop_opencv()
        else:
            run_web_dashboard(auto_open=not args.no_browser)
    except (KeyboardInterrupt, SystemExit):
        print("\n[GuardianDrive AI] Exited.")


if __name__ == "__main__":
    main()
