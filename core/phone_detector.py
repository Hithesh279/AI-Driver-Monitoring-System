"""
YOLOv8 Phone & Bottle Usage Detection Module.
Optimized for CPU via frame-sampling and persistent bounding box tracking.

Detects two COCO classes in one pass:
  - 'cell phone' (67)  -> phone-while-driving alert
  - 'bottle'     (39)  -> drink-while-driving alert (advises pulling over first)

Tilt/occlusion robustness (e.g. phone held at an angle against the ear):
  - Lower confidence threshold (config.YOLO_CONF_THRESHOLD) so partially
    occluded/rotated objects still register instead of being filtered out.
  - Larger inference resolution (config.YOLO_INFER_IMGSZ) improves recall on
    small or rotated objects compared to a heavily downscaled frame.
  - `augment=True` (test-time augmentation: horizontal flip + multi-scale)
    trades a little CPU time for meaningfully better recall on tilted poses.
  - More frequent sampling (config.YOLO_FRAME_INTERVAL) so a brief tilt isn't
    missed between inference frames.
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from config import (
    YOLO_MODEL_NAME,
    YOLO_PHONE_CLASS_ID,
    YOLO_BOTTLE_CLASS_ID,
    YOLO_CONF_THRESHOLD,
    YOLO_INFER_IMGSZ,
    YOLO_FRAME_INTERVAL,
    PHONE_HOLD_TIME_SEC,
    BOTTLE_HOLD_TIME_SEC,
    PHONE_PROXIMITY_RATIO,
)


class _HoldTimer:
    """Tracks how long a class has been continuously detected."""
    def __init__(self, hold_threshold_sec: float):
        self.hold_threshold_sec = hold_threshold_sec
        self.start_time: Optional[float] = None
        self.duration_sec = 0.0

    def update(self, detected: bool, now: float) -> None:
        if detected:
            if self.start_time is None:
                self.start_time = now
            self.duration_sec = now - self.start_time
        else:
            self.start_time = None
            self.duration_sec = 0.0

    @property
    def is_alert(self) -> bool:
        return self.duration_sec >= self.hold_threshold_sec


class PhoneDetector:
    def __init__(self):
        self.model = None
        self._load_model()

        self.frame_count = 0
        self.phone_timer = _HoldTimer(PHONE_HOLD_TIME_SEC)
        self.bottle_timer = _HoldTimer(BOTTLE_HOLD_TIME_SEC)

        self.last_detection: Dict[str, Any] = {
            "phone_detected": False, "confidence": 0.0, "bbox": None,
            "phone_duration_sec": 0.0, "is_phone_alert": False,
            "bottle_detected": False, "bottle_confidence": 0.0, "bottle_bbox": None,
            "bottle_duration_sec": 0.0, "is_bottle_alert": False,
        }

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL_NAME)
        except Exception as e:
            print(f"[Warning] Failed to load YOLOv8 model ({e}). Mock detector active.")
            self.model = None

    def detect(self, frame: np.ndarray, landmarks: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        self.frame_count += 1
        current_time = time.time()

        # Compute full 468-point MediaPipe face bounding box for driver interaction zone
        driver_zone = None
        if landmarks and len(landmarks) >= 33:
            try:
                face_pts = landmarks[:468] if len(landmarks) >= 468 else landmarks
                xs = [pt[0] for pt in face_pts]
                ys = [pt[1] for pt in face_pts]
                fx1, fx2 = min(xs), max(xs)
                fy1, fy2 = min(ys), max(ys)
                fw = max(10, fx2 - fx1)
                fh = max(10, fy2 - fy1)
                h, w = frame.shape[:2]
                driver_zone = (
                    max(0, fx1 - int(PHONE_PROXIMITY_RATIO * fw)),
                    max(0, fy1 - int(0.5 * fh)),
                    min(w, fx2 + int(PHONE_PROXIMITY_RATIO * fw)),
                    min(h, fy2 + int(3.0 * fh))
                )
            except Exception:
                driver_zone = None

        if (self.frame_count % YOLO_FRAME_INTERVAL == 0) and (self.model is not None):
            try:
                results = self.model(
                    frame,
                    classes=[YOLO_PHONE_CLASS_ID, YOLO_BOTTLE_CLASS_ID],
                    conf=YOLO_CONF_THRESHOLD,
                    verbose=False,
                    imgsz=YOLO_INFER_IMGSZ,
                    augment=True,   # test-time augmentation: better recall on tilted/partial objects
                )

                phone_found, bottle_found = False, False
                phone_conf, bottle_conf = 0.0, 0.0
                phone_bbox, bottle_bbox = None, None

                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if conf < YOLO_CONF_THRESHOLD:
                            continue
                        coords = box.xyxy[0].cpu().numpy()
                        bbox = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))

                        # Verify bounding box is in driver proximity zone to avoid passenger false positives
                        if driver_zone is not None:
                            bx_center = (bbox[0] + bbox[2]) / 2.0
                            by_center = (bbox[1] + bbox[3]) / 2.0
                            if not (driver_zone[0] <= bx_center <= driver_zone[2] and
                                    driver_zone[1] <= by_center <= driver_zone[3]):
                                continue

                        if cls_id == YOLO_PHONE_CLASS_ID:
                            phone_found = True
                            if conf > phone_conf:
                                phone_conf, phone_bbox = conf, bbox
                        elif cls_id == YOLO_BOTTLE_CLASS_ID:
                            bottle_found = True
                            if conf > bottle_conf:
                                bottle_conf, bottle_bbox = conf, bbox

                self.phone_timer.update(phone_found, current_time)
                self.bottle_timer.update(bottle_found, current_time)

                self.last_detection = {
                    "phone_detected": phone_found,
                    "confidence": float(phone_conf),
                    "bbox": phone_bbox,
                    "phone_duration_sec": float(self.phone_timer.duration_sec),
                    "is_phone_alert": self.phone_timer.is_alert,
                    "bottle_detected": bottle_found,
                    "bottle_confidence": float(bottle_conf),
                    "bottle_bbox": bottle_bbox,
                    "bottle_duration_sec": float(self.bottle_timer.duration_sec),
                    "is_bottle_alert": self.bottle_timer.is_alert,
                }
            except Exception as e:
                print(f"[Error in YOLO detection]: {e}")
        else:
            # Keep duration timers ticking between sampled frames
            if self.last_detection["phone_detected"]:
                self.phone_timer.update(True, current_time)
                self.last_detection["phone_duration_sec"] = float(self.phone_timer.duration_sec)
                self.last_detection["is_phone_alert"] = self.phone_timer.is_alert
            if self.last_detection["bottle_detected"]:
                self.bottle_timer.update(True, current_time)
                self.last_detection["bottle_duration_sec"] = float(self.bottle_timer.duration_sec)
                self.last_detection["is_bottle_alert"] = self.bottle_timer.is_alert

        return self.last_detection
