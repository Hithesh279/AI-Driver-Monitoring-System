"""
Camera and Video Capture Handler.
Streams live video frames directly from physical camera hardware (webcams, USB cameras, telematics).
Strictly uses 100% live hardware video — no synthetic cartoon frames.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import time
import threading
import numpy as np
from typing import Optional, Union

try:
    from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
except ImportError:
    from ..config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS


class CameraStream:
    """
    Threaded live camera stream handler.
    Captures live video frames directly from the physical webcam.
    """

    def __init__(self, source: Optional[Union[int, str]] = None):
        self.source = source if source is not None else CAMERA_INDEX
        self.cap = None
        self.thread = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.fps = TARGET_FPS
        self.is_live = False

        self._init_camera()

    def _init_camera(self):
        """Attempts to open physical webcam device with fallback to available indices."""
        indices_to_try = [self.source]
        if isinstance(self.source, int):
            for alt in [0, 1]:
                if alt not in indices_to_try:
                    indices_to_try.append(alt)

        for idx in indices_to_try:
            try:
                # On Windows, try default or CAP_DSHOW for faster direct hardware capture
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.source = idx
                        self.is_live = True
                        self.current_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                        print(f"[CameraStream] Connected to physical camera hardware (index {idx})")
                        return
                    else:
                        cap.release()
            except Exception as e:
                print(f"[CameraStream] Error querying camera index {idx}: {e}")

        print("[CameraStream] Warning: Physical camera device not yet accessible. Waiting for video source...")
        self.is_live = False
        self.current_frame = self._generate_standby_frame("Searching for live camera device...")

    def _generate_standby_frame(self, message: str) -> np.ndarray:
        """Clean dark standby placeholder shown only while hardware camera is connecting."""
        img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        img[:] = (20, 24, 30)
        cv2.putText(img, "GUARDIANDRIVE AI - LIVE FEED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
        cv2.putText(img, message, (20, FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(img, f"Connecting to physical webcam device (source {self.source})...",
                    (20, (FRAME_HEIGHT // 2) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 150, 165), 1)
        return img

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Continuously reads live frames from physical webcam hardware."""
        last_reconnect_time = 0.0

        while self.is_running:
            start_time = time.time()

            if self.cap is not None and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                        with self.lock:
                            self.current_frame = frame_resized
                        self.is_live = True
                    else:
                        # Frame read failed — schedule physical reconnect
                        self.is_live = False
                        if (start_time - last_reconnect_time) > 2.0:
                            self._reconnect_camera()
                            last_reconnect_time = start_time
                except Exception:
                    self.is_live = False
            else:
                # Attempt to open physical camera every 2 seconds
                if (start_time - last_reconnect_time) > 2.0:
                    self._reconnect_camera()
                    last_reconnect_time = start_time

            # Cap frame rate to TARGET_FPS
            elapsed = time.time() - start_time
            sleep_time = max(0.001, (1.0 / self.fps) - elapsed)
            time.sleep(sleep_time)

    def _reconnect_camera(self):
        """Re-initializes physical camera hardware."""
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.is_live = True
                    with self.lock:
                        self.current_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                    print(f"[CameraStream] Reconnected to live camera hardware (source {self.source})")
        except Exception:
            pass

    def set_external_frame(self, frame: np.ndarray):
        """Allows feeding live video frames directly from browser webcam stream."""
        if frame is not None:
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            with self.lock:
                self.current_frame = resized
                self.is_live = True

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def stop(self):
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


if __name__ == "__main__":
    print("[CameraStream] Testing live camera hardware...")
    stream = CameraStream()
    stream.start()
    try:
        for i in range(5):
            time.sleep(0.2)
            frame = stream.get_frame()
            if frame is not None:
                print(f"[CameraStream] Frame {i+1} received: shape={frame.shape}, is_live={stream.is_live}")
    finally:
        stream.stop()
        print("[CameraStream] Camera test finished successfully.")
