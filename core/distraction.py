"""
Distraction Detection Module:
Head Pose Estimation using solvePnP (Yaw, Pitch, Roll) and Iris Gaze Tracking.
"""

import cv2
import numpy as np
import time
from collections import deque
from typing import List, Tuple, Dict, Any
from config import (
    YAW_LIMIT_DEG,
    PITCH_LIMIT_DEG,
    ROLL_LIMIT_DEG,
    GAZE_DEVIATION_THRESHOLD,
    DISTRACTION_TIME_SEC,
    TARGET_FPS,
    SMOOTHING_ALPHA
)

# 3D Model Reference Points (Generic 3D Human Face Model in millimeters, OpenCV Camera Coordinates: X-Right, Y-Down, Z-Forward)
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
    (0.0, 330.0, -65.0),         # Chin (Landmark 152)
    (-225.0, -170.0, -135.0),    # Left Eye Left Corner (Landmark 33)
    (225.0, -170.0, -135.0),     # Right Eye Right Corner (Landmark 263)
    (-150.0, 150.0, -125.0),     # Left Mouth Corner (Landmark 61)
    (150.0, 150.0, -125.0)       # Right Mouth Corner (Landmark 291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
LANDMARK_POSE_INDICES = [1, 152, 33, 263, 61, 291]


class DistractionDetector:
    def __init__(self, fps: int = TARGET_FPS):
        self.fps = fps
        self.consec_distracted_frames = 0
        self.is_distracted = False
        self.distraction_start_time = None
        self.distraction_duration_sec = 0.0
        self.pose_history = deque(maxlen=fps * 10)  # 10s sliding window for head-pose variance
        
        # Calibration baseline offsets (calibrated at startup if requested)
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_roll = 0.0
        self.alpha = SMOOTHING_ALPHA
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_roll = 0.0

    def estimate_head_pose(self, landmarks: List[Tuple[int, int]], frame_shape: Tuple[int, int, int]) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Solves Perspective-n-Point (solvePnP) to determine 3D orientation (Yaw, Pitch, Roll).
        Angles are computed accurately in degrees.
        """
        h, w, _ = frame_shape
        
        # 2D Image Points
        image_points = np.array([
            landmarks[idx] for idx in LANDMARK_POSE_INDICES
        ], dtype=np.float64)

        # Approximate Camera Intrinsic Matrix
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1)) # Assuming minimal lens distortion

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles (cv2.RQDecomp3x3 returns angles directly in degrees)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Normalize roll to [-90, 90]
        if abs(roll) > 90.0:
            roll = roll - 180.0 if roll > 0.0 else roll + 180.0

        # Adjust for calibration offsets
        adj_pitch = pitch - self.baseline_pitch
        adj_yaw = yaw - self.baseline_yaw
        adj_roll = roll - self.baseline_roll

        # Apply EMA temporal smoothing to eliminate sensor jitter
        self.smoothed_yaw = self.alpha * adj_yaw + (1.0 - self.alpha) * self.smoothed_yaw
        self.smoothed_pitch = self.alpha * adj_pitch + (1.0 - self.alpha) * self.smoothed_pitch
        self.smoothed_roll = self.alpha * adj_roll + (1.0 - self.alpha) * self.smoothed_roll

        return float(self.smoothed_yaw), float(self.smoothed_pitch), float(self.smoothed_roll), rvec, tvec

    def estimate_gaze_direction(self, landmarks: List[Tuple[int, int]]) -> Tuple[str, float]:
        """
        Calculates relative iris displacement to estimate 2D gaze direction (horizontal & vertical).
        MediaPipe Refined landmarks:
        - Left eye corners: 33, 133. Left iris center: 468. Top/bottom lid: 159, 145
        - Right eye corners: 362, 263. Right iris center: 473. Top/bottom lid: 386, 374
        """
        try:
            if len(landmarks) < 478:
                return "Forward", 0.0

            # Horizontal Gaze (Left / Right)
            l_corner_outer = landmarks[33]
            l_corner_inner = landmarks[133]
            l_iris = landmarks[468]

            l_width = max(1e-6, abs(l_corner_inner[0] - l_corner_outer[0]))
            l_ratio = (l_iris[0] - min(l_corner_outer[0], l_corner_inner[0])) / float(l_width)

            r_corner_inner = landmarks[362]
            r_corner_outer = landmarks[263]
            r_iris = landmarks[473]

            r_width = max(1e-6, abs(r_corner_outer[0] - r_corner_inner[0]))
            r_ratio = (r_iris[0] - min(r_corner_inner[0], r_corner_outer[0])) / float(r_width)

            avg_ratio = (l_ratio + r_ratio) / 2.0

            # Vertical Gaze (Looking Down / Up)
            l_top = landmarks[159]
            l_bot = landmarks[145]
            l_eye_h = max(1e-6, abs(l_bot[1] - l_top[1]))
            l_v_ratio = (l_iris[1] - l_top[1]) / float(l_eye_h)

            r_top = landmarks[386]
            r_bot = landmarks[374]
            r_eye_h = max(1e-6, abs(r_bot[1] - r_top[1]))
            r_v_ratio = (r_iris[1] - r_top[1]) / float(r_eye_h)

            avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

            # Suppress gaze calculation when eyes are closed or in mid-blink
            # (prevents false 'Looking Down' distraction alerts caused by vertical eyelid collapse)
            l_aperture = l_eye_h / float(l_width)
            r_aperture = r_eye_h / float(r_width)
            eyes_open = (l_aperture >= 0.12) and (r_aperture >= 0.12) and (l_eye_h >= 3.0) and (r_eye_h >= 3.0)

            # Calibrated directions
            if eyes_open and avg_v_ratio > 0.72:
                direction = "Looking Down"
                deviation = abs(avg_v_ratio - 0.50)
            elif eyes_open and avg_ratio < 0.38:
                direction = "Right"
                deviation = abs(0.50 - avg_ratio)
            elif eyes_open and avg_ratio > 0.62:
                direction = "Left"
                deviation = abs(avg_ratio - 0.50)
            else:
                direction = "Forward"
                deviation = 0.0

            return direction, float(deviation)
        except Exception:
            return "Forward", 0.0

    def update(self, landmarks: List[Tuple[int, int]], frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Updates distraction metrics based on head pose angles and gaze.
        """
        yaw, pitch, roll, rvec, tvec = self.estimate_head_pose(landmarks, frame_shape)
        gaze_dir, gaze_dev = self.estimate_gaze_direction(landmarks)

        # Check off-road triggers (Yaw left/right, Pitch up/down, Roll sideways tilt)
        head_off_road = (abs(yaw) > YAW_LIMIT_DEG) or (abs(pitch) > PITCH_LIMIT_DEG) or (abs(roll) > ROLL_LIMIT_DEG)
        gaze_off_road = (gaze_dir != "Forward") and (gaze_dev >= GAZE_DEVIATION_THRESHOLD)

        is_off_road_now = head_off_road or gaze_off_road

        if is_off_road_now:
            self.consec_distracted_frames += 1
            if self.distraction_start_time is None:
                self.distraction_start_time = time.time()
            self.distraction_duration_sec = time.time() - self.distraction_start_time
        else:
            self.consec_distracted_frames = 0
            self.distraction_start_time = None
            self.distraction_duration_sec = 0.0

        # Record in sliding window and calculate head pose variance (PPT Slide 10)
        self.pose_history.append((float(yaw), float(pitch), float(roll)))
        if len(self.pose_history) >= 2:
            y_arr = [p[0] for p in self.pose_history]
            p_arr = [p[1] for p in self.pose_history]
            head_pose_variance = float(np.var(y_arr) + np.var(p_arr))
        else:
            head_pose_variance = 0.0

        # Alert if distraction exceeds threshold time
        self.is_distracted = self.distraction_duration_sec >= DISTRACTION_TIME_SEC

        distraction_type = "NONE"
        distraction_reason = "Normal forward attention"
        if self.is_distracted:
            if pitch < -16.0 or gaze_dir == "Looking Down":
                distraction_type = "LOOKING_DOWN"
                distraction_reason = "Driver looking down at lap or console"
            elif yaw < -20.0 or (gaze_dir == "Left" and yaw < -8.0):
                distraction_type = "LOOKING_LEFT"
                distraction_reason = "Driver looking left away from road"
            elif yaw > 20.0 or (gaze_dir == "Right" and yaw > 8.0):
                distraction_type = "LOOKING_RIGHT"
                distraction_reason = "Driver looking right away from road"
            elif pitch > 18.0:
                distraction_type = "LOOKING_UP"
                distraction_reason = "Driver head tilted up away from road"
            elif abs(roll) > 24.0:
                distraction_type = "HEAD_TILT"
                distraction_reason = "Driver head tilted sideways / slumping posture"
            elif gaze_off_road:
                distraction_type = "GAZE_OFF_ROAD"
                distraction_reason = "Driver gaze wandering away from road"
            else:
                distraction_type = "GENERAL_OFF_ROAD"
                distraction_reason = "Driver attention diverted away from road"

        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "head_pose_variance": round(head_pose_variance, 2),
            "gaze_direction": gaze_dir,
            "gaze_deviation": float(gaze_dev),
            "head_off_road": head_off_road,
            "gaze_off_road": gaze_off_road,
            "distraction_duration_sec": float(self.distraction_duration_sec),
            "is_distracted": self.is_distracted,
            "distraction_type": distraction_type,
            "distraction_reason": distraction_reason,
            "rvec": rvec,
            "tvec": tvec
        }

    def calibrate(self, current_yaw: float, current_pitch: float, current_roll: float):
        """Sets current head position as the baseline forward-facing orientation and resets smoothed state."""
        self.baseline_yaw += current_yaw
        self.baseline_pitch += current_pitch
        self.baseline_roll += current_roll
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_roll = 0.0
        self.consec_distracted_frames = 0
        self.distraction_start_time = None
        self.distraction_duration_sec = 0.0
        self.is_distracted = False
