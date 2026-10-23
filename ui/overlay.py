"""
OpenCV Real-Time HUD Overlay Renderer.
Visualizes facial mesh, 3D head pose axes, phone & bottle bounding boxes, telemetry gauges, and alert banners.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from config import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    LEFT_IRIS_LANDMARKS,
    RIGHT_IRIS_LANDMARKS,
    LIPS_LANDMARKS
)

def draw_hud_overlay(frame: np.ndarray,
                     landmarks: Optional[List[Tuple[int, int]]],
                     drowsiness_data: Dict[str, Any],
                     distraction_data: Dict[str, Any],
                     phone_data: Dict[str, Any],
                     risk_data: Dict[str, Any],
                     fps: float = 30.0) -> np.ndarray:
    """
    Renders telemetry, annotations, and visual safety badges onto the video frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # -------------------------------------------------------------
    # 1. Draw Facial Mesh Contours & Iris Points
    # -------------------------------------------------------------
    if landmarks is not None:
        # Draw eyes contour
        for pts_indices, color in [(LEFT_EYE_LANDMARKS, (0, 255, 200)), (RIGHT_EYE_LANDMARKS, (0, 255, 200))]:
            pts = np.array([landmarks[i] for i in pts_indices], np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)

        # Draw irises
        for iris_indices in [LEFT_IRIS_LANDMARKS, RIGHT_IRIS_LANDMARKS]:
            if len(landmarks) >= 478:
                for idx in iris_indices:
                    cv2.circle(overlay, landmarks[idx], 1, (0, 255, 255), -1)

        # Draw lips contour
        lip_pts = np.array([landmarks[i] for i in LIPS_LANDMARKS], np.int32)
        cv2.polylines(overlay, [lip_pts], isClosed=False, color=(255, 120, 255), thickness=1)

        # Draw 3D Head Pose Orientation Axes
        rvec = distraction_data.get("rvec")
        tvec = distraction_data.get("tvec")
        if rvec is not None and tvec is not None and np.any(rvec):
            nose_pt = landmarks[1]
            axis_length = 60.0
            
            # 3D axis points
            axis_3d = np.array([
                (axis_length, 0.0, 0.0),   # X - Red
                (0.0, axis_length, 0.0),   # Y - Green
                (0.0, 0.0, -axis_length)   # Z - Blue (pointing forward from face)
            ], dtype=np.float64)

            # Camera intrinsics
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            proj_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
            p_nose = (int(nose_pt[0]), int(nose_pt[1]))
            p_x = (int(proj_pts[0][0][0]), int(proj_pts[0][0][1]))
            p_y = (int(proj_pts[1][0][0]), int(proj_pts[1][0][1]))
            p_z = (int(proj_pts[2][0][0]), int(proj_pts[2][0][1]))

            cv2.line(overlay, p_nose, p_x, (0, 0, 255), 2) # X Axis (Red)
            cv2.line(overlay, p_nose, p_y, (0, 255, 0), 2) # Y Axis (Green)
            cv2.line(overlay, p_nose, p_z, (255, 0, 0), 2) # Z Axis (Blue)

    # -------------------------------------------------------------
    # 2. Draw Phone & Bottle Bounding Boxes
    # -------------------------------------------------------------
    if phone_data.get("phone_detected", False) and phone_data.get("bbox") is not None:
        x1, y1, x2, y2 = phone_data["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Phone: {int(phone_data.get('confidence', 0)*100)}%"
        cv2.putText(overlay, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if phone_data.get("bottle_detected", False) and phone_data.get("bottle_bbox") is not None:
        bx1, by1, bx2, by2 = phone_data["bottle_bbox"]
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
        blabel = f"Bottle: {int(phone_data.get('bottle_confidence', 0)*100)}%"
        cv2.putText(overlay, blabel, (bx1, max(15, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # -------------------------------------------------------------
    # 3. Top Risk Banner
    # -------------------------------------------------------------
    risk_level = risk_data.get("level", "Low")
    risk_score = risk_data.get("score", 0.0)
    
    if risk_level == "High":
        banner_bg = (0, 0, 180) # Dark red
        badge_text = f"CRITICAL RISK: {risk_score:.2f}"
    elif risk_level == "Medium":
        banner_bg = (0, 140, 200) # Amber
        badge_text = f"CAUTION RISK: {risk_score:.2f}"
    else:
        banner_bg = (20, 140, 40) # Green
        badge_text = f"ATTENTIVE: {risk_score:.2f}"

    # Header bar
    cv2.rectangle(overlay, (0, 0), (w, 38), banner_bg, -1)
    cv2.putText(overlay, badge_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
    # FPS Counter
    cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # 4. Bottom Telemetry Bar
    # -------------------------------------------------------------
    cv2.rectangle(overlay, (0, h - 34), (w, h), (15, 15, 15), -1)
    
    ear = drowsiness_data.get("ear", 0.0)
    mar = drowsiness_data.get("mar", 0.0)
    perclos = drowsiness_data.get("perclos", 0.0) * 100
    yaw = distraction_data.get("yaw", 0.0)
    pitch = distraction_data.get("pitch", 0.0)
    phone_str = "YES" if phone_data.get("phone_detected", False) else "NO"

    telemetry_str = f"EAR:{ear:.2f} | MAR:{mar:.2f} | PERCLOS:{perclos:.0f}% | YAW:{yaw:+.0f}° | PHONE:{phone_str}"
    cv2.putText(overlay, telemetry_str, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # 5. Flashing Specific Alert Popup Banner (matching exact mistake)
    # -------------------------------------------------------------
    if drowsiness_data.get("is_microsleep", False):
        _draw_flashing_alert(overlay, "CRITICAL: EYES CLOSED! WAKE UP NOW!", w, h)
    elif phone_data.get("is_phone_alert", False) or (phone_data.get("phone_detected", False) and phone_data.get("phone_duration_sec", 0.0) >= 0.5):
        _draw_flashing_alert(overlay, "DON'T USE THE PHONE! KEEP YOUR EYES ON THE ROAD!", w, h)
    elif phone_data.get("is_bottle_alert", False):
        _draw_flashing_alert(overlay, "ALERT: DRINKING DETECTED - PULL OVER SAFELY!", w, h)
    elif distraction_data.get("is_distracted", False):
        dtype = distraction_data.get("distraction_type", "")
        pitch = distraction_data.get("pitch", 0.0)
        yaw = distraction_data.get("yaw", 0.0)
        roll = distraction_data.get("roll", 0.0)
        gaze_dir = distraction_data.get("gaze_direction", "Forward")

        if dtype == "LOOKING_DOWN" or pitch < -16.0 or gaze_dir == "Looking Down":
            _draw_flashing_alert(overlay, "ALERT: LOOKING DOWN - EYES UP & WATCH ROAD!", w, h)
        elif dtype == "LOOKING_LEFT" or yaw < -20.0 or (gaze_dir == "Left" and yaw < -8.0):
            _draw_flashing_alert(overlay, "ALERT: LOOKING LEFT - FACE FORWARD & WATCH ROAD!", w, h)
        elif dtype == "LOOKING_RIGHT" or yaw > 20.0 or (gaze_dir == "Right" and yaw > 8.0):
            _draw_flashing_alert(overlay, "ALERT: LOOKING RIGHT - FACE FORWARD & WATCH ROAD!", w, h)
        elif dtype == "LOOKING_UP" or pitch > 18.0:
            _draw_flashing_alert(overlay, "ALERT: LOOKING UP - KEEP EYES LEVEL ON ROAD!", w, h)
        elif dtype == "HEAD_TILT" or abs(roll) > 24.0:
            _draw_flashing_alert(overlay, "ALERT: HEAD TILT - SIT UPRIGHT & FOCUS AHEAD!", w, h)
        elif dtype == "GAZE_OFF_ROAD" or distraction_data.get("gaze_off_road", False):
            _draw_flashing_alert(overlay, "ALERT: EYES OFF ROAD - FOCUS ATTENTION FORWARD!", w, h)
        else:
            _draw_flashing_alert(overlay, "ALERT: DISTRACTED - KEEP EYES ON ROAD!", w, h)
    elif drowsiness_data.get("is_drowsy", False) and drowsiness_data.get("closure_duration_sec", 0.0) >= 0.8:
        _draw_flashing_alert(overlay, "WARNING: EYES DROOPING - STAY ALERT!", w, h)
    elif drowsiness_data.get("is_yawning", False):
        _draw_flashing_alert(overlay, "CAUTION: YAWN DETECTED - REST BREAK RECOMMENDED!", w, h)

    # Blend overlay with slight alpha for smooth appearance
    alpha = 0.92
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def _draw_flashing_alert(img: np.ndarray, text: str, w: int, h: int):
    """Draws high-visibility center warning banner."""
    box_h = 38
    y_start = (h // 2) - (box_h // 2)
    cv2.rectangle(img, (14, y_start), (w - 14, y_start + box_h), (0, 0, 220), -1)
    cv2.rectangle(img, (14, y_start), (w - 14, y_start + box_h), (255, 255, 255), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    tx = max(18, (w - tw) // 2)
    cv2.putText(img, text, (tx, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
