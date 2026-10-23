"""
Drowsiness and Fatigue Detection Module:
Calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), rolling PERCLOS, and Blink Rate.
"""

import time
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any
from config import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    MAR_THRESHOLD,
    MAR_CONSEC_FRAMES,
    YAWN_COOLDOWN_SEC,
    PERCLOS_WINDOW_SECONDS,
    PERCLOS_FATIGUE_THRESHOLD,
    MICROSLEEP_TIME_SEC,
    TARGET_FPS
)

def euclidean_dist(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    """Calculates Euclidean distance between two 2D points."""
    return float(np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2))

def calculate_ear(landmarks: List[Tuple[int, int]], eye_indices: List[int]) -> float:
    """
    Computes Eye Aspect Ratio (EAR) using 6 landmark points:
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    try:
        p1 = landmarks[eye_indices[0]] # Outer corner
        p2 = landmarks[eye_indices[1]] # Top outer
        p3 = landmarks[eye_indices[2]] # Top inner
        p4 = landmarks[eye_indices[3]] # Inner corner
        p5 = landmarks[eye_indices[4]] # Bottom inner
        p6 = landmarks[eye_indices[5]] # Bottom outer

        v1 = euclidean_dist(p2, p6)
        v2 = euclidean_dist(p3, p5)
        h = euclidean_dist(p1, p4)

        if h < 1e-6:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    except IndexError:
        return 0.0

def calculate_mar(landmarks: List[Tuple[int, int]]) -> float:
    """
    Computes Mouth Aspect Ratio (MAR) to detect yawning:
    Uses inner lip vertical distances divided by mouth horizontal width.
    """
    try:
        # Landmarks: 61 (left corner), 291 (right corner), 13 (upper inner lip), 14 (lower inner lip)
        # 0 (upper outer lip), 17 (lower outer lip), 78 (left inner), 308 (right inner)
        left = landmarks[61]
        right = landmarks[291]
        top_inner = landmarks[13]
        bot_inner = landmarks[14]
        top_outer = landmarks[0]
        bot_outer = landmarks[17]

        h = euclidean_dist(left, right)
        if h < 1e-6:
            return 0.0

        v1 = euclidean_dist(top_inner, bot_inner)
        v2 = euclidean_dist(top_outer, bot_outer)
        
        # Weighted average vertical distance divided by horizontal width
        mar = (v1 * 1.5 + v2 * 0.5) / (2.0 * h)
        return float(mar)
    except IndexError:
        return 0.0


class DrowsinessDetector:
    def __init__(self, fps: int = TARGET_FPS):
        self.fps = fps
        self.max_window_size = int(fps * PERCLOS_WINDOW_SECONDS)
        
        # State tracking
        self.ear_history = deque(maxlen=self.max_window_size)
        self.eye_closed_history = deque(maxlen=self.max_window_size)
        self.timestamps = deque(maxlen=self.max_window_size)
        
        # Consecutive frame counters & timestamps
        self.consec_closed_frames = 0
        self.consec_yawn_frames = 0
        self.eye_closed_start_time = 0.0
        
        # Blink tracking
        self.blink_count = 0
        self.is_blinking = False
        self.blink_timestamps = deque(maxlen=200)
        
        # Yawn tracking
        self.total_yawns = 0
        self.is_yawning = False
        self.last_yawn_time = 0.0
        self.yawn_history = deque(maxlen=60) # Timestamps of recent yawns
        self.baseline_mar = 0.20 # Adaptive resting mouth baseline
        self.baseline_ear = 0.28 # Adaptive resting eye aperture baseline
        self.alpha = 0.65        # EMA smoothing factor for anti-jitter
        self.smoothed_ear = 0.28
        self.smoothed_mar = 0.20
        self.is_calibrated = False

    def calibrate(self, baseline_ear: float = None, baseline_mar: float = None):
        """Manually sets the driver's unique resting baseline EAR and MAR."""
        if baseline_ear is not None and baseline_ear > 0.15:
            self.baseline_ear = float(baseline_ear)
            self.smoothed_ear = float(baseline_ear)
        if baseline_mar is not None and baseline_mar > 0.05:
            self.baseline_mar = float(baseline_mar)
            self.smoothed_mar = float(baseline_mar)
        self.is_calibrated = True

    def update(self, landmarks: List[Tuple[int, int]]) -> Dict[str, Any]:
        current_time = time.time()
        
        # 1. Calculate EAR for left and right eyes
        left_ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_LANDMARKS)
        raw_ear = (left_ear + right_ear) / 2.0
        
        # Apply Temporal EMA smoothing to filter sensor noise / jitter
        self.smoothed_ear = self.alpha * raw_ear + (1.0 - self.alpha) * self.smoothed_ear
        ear = self.smoothed_ear
        
        # 2. Calculate MAR for mouth opening with EMA smoothing
        raw_mar = calculate_mar(landmarks)
        self.smoothed_mar = self.alpha * raw_mar + (1.0 - self.alpha) * self.smoothed_mar
        mar = self.smoothed_mar
        
        # 3. Eye closure & Blink Logic with Adaptive Baseline
        if ear >= 0.18 and not self.is_calibrated:
            self.baseline_ear = 0.99 * self.baseline_ear + 0.01 * ear

        effective_ear_thresh = min(EAR_THRESHOLD, max(0.16, self.baseline_ear * 0.72))
        is_eye_closed = ear < effective_ear_thresh
        self.ear_history.append(ear)
        self.eye_closed_history.append(1 if is_eye_closed else 0)
        self.timestamps.append(current_time)
        
        if is_eye_closed:
            self.consec_closed_frames += 1
            if not self.is_blinking:
                self.is_blinking = True
            if self.eye_closed_start_time == 0.0:
                self.eye_closed_start_time = current_time
        else:
            if self.is_blinking and self.consec_closed_frames >= 2 and self.consec_closed_frames < (EAR_CONSEC_FRAMES * 2):
                # Valid complete blink
                self.blink_count += 1
                self.blink_timestamps.append(current_time)
            self.is_blinking = False
            self.consec_closed_frames = 0
            self.eye_closed_start_time = 0.0

        # Time-based microsleep and drowsiness duration (robust against camera FPS drops)
        if is_eye_closed and self.eye_closed_start_time > 0.0:
            elapsed_sec = current_time - self.eye_closed_start_time
            closure_duration_sec = max(elapsed_sec, self.consec_closed_frames / float(self.fps))
        else:
            closure_duration_sec = 0.0

        is_microsleep = closure_duration_sec >= MICROSLEEP_TIME_SEC
        is_drowsy = (closure_duration_sec >= (EAR_CONSEC_FRAMES / float(self.fps))) or (self.consec_closed_frames >= EAR_CONSEC_FRAMES)

        # 4. Yawn Logic with Adaptive Baseline
        adaptive_yawn_thresh = max(0.38, self.baseline_mar * 1.85)
        is_mouth_open = (mar >= MAR_THRESHOLD) or (mar >= adaptive_yawn_thresh)

        if is_mouth_open:
            self.consec_yawn_frames += 1
            if self.consec_yawn_frames >= MAR_CONSEC_FRAMES and not self.is_yawning:
                if (current_time - self.last_yawn_time) > YAWN_COOLDOWN_SEC:
                    self.total_yawns += 1
                    self.is_yawning = True
                    self.last_yawn_time = current_time
                    self.yawn_history.append(current_time)
        else:
            # Smoothly update resting baseline MAR when mouth is closed
            self.baseline_mar = 0.98 * self.baseline_mar + 0.02 * mar
            self.consec_yawn_frames = 0
            self.is_yawning = False

        # 5. Calculate PERCLOS (% of time eyes closed in rolling window)
        # Requires at least 3.0 seconds (~90 frames) observation window before computing rolling ratio
        min_obs_frames = int(self.fps * 3.0)
        if len(self.eye_closed_history) >= min_obs_frames:
            perclos = sum(self.eye_closed_history) / float(len(self.eye_closed_history))
        else:
            perclos = 0.0
        
        is_fatigued = perclos >= PERCLOS_FATIGUE_THRESHOLD

        # 6. Calculate Blink Rate (blinks per minute)
        cutoff_time = current_time - 60.0
        recent_blinks = [t for t in self.blink_timestamps if t >= cutoff_time]
        blink_rate_bpm = len(recent_blinks)

        # 7. Recent Yawns in last 3 minutes
        recent_yawns_count = len([t for t in self.yawn_history if t >= (current_time - 180.0)])

        return {
            "ear": float(ear),
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "mar": float(mar),
            "perclos": float(perclos),
            "is_eye_closed": is_eye_closed,
            "closure_duration_sec": float(closure_duration_sec),
            "is_drowsy": is_drowsy,
            "is_microsleep": is_microsleep,
            "is_yawning": self.is_yawning,
            "total_yawns": self.total_yawns,
            "recent_yawns": recent_yawns_count,
            "is_fatigued": is_fatigued,
            "blink_count": self.blink_count,
            "blink_rate_bpm": blink_rate_bpm
        }
