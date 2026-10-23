"""
Emotion & Stress Recognition Module (PPT Module 03):
Classifies driver facial expressions and estimates stress / mood levels
using Facial Action Coding System (FACS) geometric landmark analysis.
Recognizes: Calm / Neutral, Stressed / Anxious, Fatigued, Frustrated / Angry.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    from config import STRESS_THRESHOLD, SMOOTHING_ALPHA, EMOTION_FRAME_INTERVAL
except ImportError:
    from ..config import STRESS_THRESHOLD, SMOOTHING_ALPHA, EMOTION_FRAME_INTERVAL


def _dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


class EmotionDetector:
    """
    Real-time Emotion and Psychological Stress Recognizer.
    Analyzes eyebrow compression, eyelid tension, mouth curvature, and jaw posture
    to classify driver state into Calm, Stressed, Fatigued, or Frustrated.
    """

    def __init__(self):
        self.frame_count = 0
        self.smoothed_stress = 0.10
        self.alpha = SMOOTHING_ALPHA

        # Baseline facial measurements (calibrated over time)
        self.baseline_brow_dist = None
        self.baseline_mouth_width = None

        self.last_result: Dict[str, Any] = {
            "dominant_emotion": "Calm",
            "stress_score": 0.10,
            "is_stressed": False,
            "probabilities": {
                "Calm": 0.85,
                "Stressed": 0.05,
                "Fatigued": 0.05,
                "Frustrated": 0.05,
            },
            "brow_furrow_index": 0.0,
            "mouth_tension_index": 0.0,
        }

    def update(
        self,
        landmarks: Optional[List[Tuple[int, int]]],
        drowsiness_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.frame_count += 1

        if landmarks is None or len(landmarks) < 468:
            return self.last_result

        # Sample every N frames to save CPU while maintaining responsiveness
        if (self.frame_count % EMOTION_FRAME_INTERVAL) != 0:
            return self.last_result

        try:
            # Key facial feature anchors
            # Outer eye corners for face scale normalization: left 33, right 263
            eye_scale = max(1e-5, _dist(landmarks[33], landmarks[263]))

            # Inner eyebrow landmarks: left 107 (inner right of driver), right 336 (inner left)
            inner_brow_dist = _dist(landmarks[107], landmarks[336]) / eye_scale

            # Eye tops to inner eyebrows (height of eyebrows above eyes)
            # Left brow 66 to left eye top 159, right brow 296 to right eye top 386
            brow_eye_dist = (
                _dist(landmarks[66], landmarks[159]) + _dist(landmarks[296], landmarks[386])
            ) / (2.0 * eye_scale)

            # Mouth corners (61, 291) and lips (0 top outer, 17 bottom outer, 13 top inner, 14 bottom inner)
            mouth_width = _dist(landmarks[61], landmarks[291]) / eye_scale
            mouth_corners_y = (landmarks[61][1] + landmarks[291][1]) / 2.0
            lip_center_y = landmarks[0][1]
            mouth_curvature = (mouth_corners_y - lip_center_y) / eye_scale  # positive = smiling, negative = frowning

            # Resting baseline initialization
            if self.baseline_brow_dist is None:
                self.baseline_brow_dist = inner_brow_dist
                self.baseline_mouth_width = mouth_width
            else:
                # Slowly adapt baseline to driver's resting neutral posture
                self.baseline_brow_dist = 0.995 * self.baseline_brow_dist + 0.005 * inner_brow_dist
                self.baseline_mouth_width = 0.995 * self.baseline_mouth_width + 0.005 * mouth_width

            # 1. Brow Furrow Index (Corrugator supercilii contraction = anxiety / frustration)
            # Smaller distance between inner eyebrows = tighter furrowing
            furrow_ratio = self.baseline_brow_dist / max(1e-5, inner_brow_dist)
            brow_furrow = float(np.clip((furrow_ratio - 1.0) * 3.5, 0.0, 1.0))

            # 2. Mouth Tension Index (lip compression or frown grimace)
            mouth_tension = float(np.clip(abs(mouth_curvature - 0.08) * 4.0, 0.0, 1.0))

            # 3. Incorporate fatigue cues from drowsiness detector if available
            is_fatigued = False
            perclos = 0.0
            if drowsiness_data:
                is_fatigued = drowsiness_data.get("is_fatigued", False) or drowsiness_data.get("is_drowsy", False)
                perclos = drowsiness_data.get("perclos", 0.0)

            # 4. Multi-class Emotion Likelihood Estimation
            # Raw unnormalized weights
            score_frustrated = 0.60 * brow_furrow + 0.40 * mouth_tension
            score_stressed = 0.50 * brow_furrow + 0.30 * (1.0 - min(1.0, brow_eye_dist * 2.0)) + 0.20 * mouth_tension
            score_fatigued = 0.70 * min(1.0, perclos * 4.0) + 0.30 * (1.0 if is_fatigued else 0.0)
            score_calm = max(0.05, 1.0 - (score_frustrated * 0.7 + score_stressed * 0.7 + score_fatigued * 0.6))

            # Softmax / probability normalization
            raw_scores = np.array([score_calm, score_stressed, score_fatigued, score_frustrated], dtype=np.float64)
            exp_scores = np.exp(raw_scores * 2.5)
            probs = exp_scores / np.sum(exp_scores)

            prob_dict = {
                "Calm": round(float(probs[0]), 3),
                "Stressed": round(float(probs[1]), 3),
                "Fatigued": round(float(probs[2]), 3),
                "Frustrated": round(float(probs[3]), 3),
            }

            class_names = ["Calm", "Stressed", "Fatigued", "Frustrated"]
            dominant = class_names[int(np.argmax(probs))]

            # Compute continuous Stress Score [0.0 - 1.0]
            instant_stress = float(np.clip(
                (prob_dict["Stressed"] * 0.65) +
                (prob_dict["Frustrated"] * 0.85) +
                (prob_dict["Fatigued"] * 0.35) +
                (brow_furrow * 0.20),
                0.0, 1.0
            ))

            # Apply Temporal EMA smoothing to prevent rapid fluttering
            self.smoothed_stress = self.alpha * instant_stress + (1.0 - self.alpha) * self.smoothed_stress
            is_stressed = self.smoothed_stress >= STRESS_THRESHOLD

            self.last_result = {
                "dominant_emotion": dominant,
                "stress_score": round(float(self.smoothed_stress), 3),
                "is_stressed": is_stressed,
                "probabilities": prob_dict,
                "brow_furrow_index": round(brow_furrow, 3),
                "mouth_tension_index": round(mouth_tension, 3),
            }
            return self.last_result

        except Exception:
            return self.last_result

    def calibrate(self, landmarks: List[Tuple[int, int]]):
        """Sets current facial expression as the neutral calm baseline."""
        if landmarks and len(landmarks) >= 468:
            eye_scale = max(1e-5, _dist(landmarks[33], landmarks[263]))
            self.baseline_brow_dist = _dist(landmarks[107], landmarks[336]) / eye_scale
            self.baseline_mouth_width = _dist(landmarks[61], landmarks[291]) / eye_scale
            self.smoothed_stress = 0.05
