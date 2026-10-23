"""
Unit tests for Computer Vision metrics and 5-Factor Risk Engine.
"""

import sys
from pathlib import Path
import numpy as np

try:
    import pytest
except ImportError:
    class _Approx:
        def __init__(self, val, abs=1e-3):
            self.val = val
            self.abs = abs
        def __eq__(self, other):
            return abs(float(other) - float(self.val)) <= self.abs
    class _PytestFallback:
        @staticmethod
        def approx(val, abs=1e-3):
            return _Approx(val, abs=abs)
    pytest = _PytestFallback()

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.drowsiness import calculate_ear, calculate_mar
from core.distraction import DistractionDetector
from core.risk_engine import RiskEngine
from config import (
    EAR_THRESHOLD,
    MAR_THRESHOLD,
    RISK_BAND_LOW_MAX,
    RISK_BAND_MED_MAX
)

def test_ear_calculation_open_and_closed():
    landmarks_open = {
        362: (0, 0),
        385: (5, 5),
        387: (15, 5),
        263: (20, 0),
        373: (15, -5),
        380: (5, -5)
    }
    landmarks_list_open = [landmarks_open.get(i, (0, 0)) for i in range(500)]
    ear_open = calculate_ear(landmarks_list_open, [362, 385, 387, 263, 373, 380])
    assert ear_open == pytest.approx(0.50, abs=0.01)
    assert ear_open > EAR_THRESHOLD

    landmarks_closed = {
        362: (0, 0),
        385: (5, 1),
        387: (15, 1),
        263: (20, 0),
        373: (15, -1),
        380: (5, -1)
    }
    landmarks_list_closed = [landmarks_closed.get(i, (0, 0)) for i in range(500)]
    ear_closed = calculate_ear(landmarks_list_closed, [362, 385, 387, 263, 373, 380])
    assert ear_closed == pytest.approx(0.10, abs=0.01)
    assert ear_closed < EAR_THRESHOLD

def test_mar_calculation_yawning():
    landmarks_neutral = {
        61: (0, 0), 291: (40, 0),
        13: (20, 4), 14: (20, -4),
        0: (20, 6), 17: (20, -6)
    }
    list_neutral = [landmarks_neutral.get(i, (0, 0)) for i in range(500)]
    mar_neutral = calculate_mar(list_neutral)
    assert mar_neutral < MAR_THRESHOLD

    landmarks_yawn = {
        61: (0, 0), 291: (40, 0),
        13: (20, 18), 14: (20, -18),
        0: (20, 22), 17: (20, -22)
    }
    list_yawn = [landmarks_yawn.get(i, (0, 0)) for i in range(500)]
    mar_yawn = calculate_mar(list_yawn)
    assert mar_yawn > MAR_THRESHOLD

def test_risk_engine_bounds_and_bands():
    engine = RiskEngine()

    # Case 1: Fully alert driver -> Low Risk
    drowsy_safe = {"ear": 0.35, "is_eye_closed": False, "closure_duration_sec": 0.0, "perclos": 0.02, "is_yawning": False, "recent_yawns": 0}
    distract_safe = {"head_off_road": False, "gaze_off_road": False, "distraction_duration_sec": 0.0}
    phone_safe = {"phone_detected": False, "confidence": 0.0, "phone_duration_sec": 0.0}
    
    res_safe = engine.evaluate(drowsy_safe, distract_safe, phone_safe)
    assert res_safe["level"] == "Low"
    assert res_safe["score"] <= RISK_BAND_LOW_MAX

    # Case 2: Extreme danger (Microsleep + Phone) -> High Risk
    drowsy_danger = {"ear": 0.05, "is_eye_closed": True, "closure_duration_sec": 2.0, "is_microsleep": True, "perclos": 0.35, "is_yawning": False, "recent_yawns": 2}
    distract_danger = {"head_off_road": True, "gaze_off_road": True, "distraction_duration_sec": 2.5, "is_distracted": True}
    phone_danger = {"phone_detected": True, "confidence": 0.92, "phone_duration_sec": 3.0, "is_phone_alert": True}

    res_danger = engine.evaluate(drowsy_danger, distract_danger, phone_danger)
    assert res_danger["level"] == "High"
    assert res_danger["score"] > RISK_BAND_MED_MAX
    assert len(res_danger["primary_reasons"]) >= 2

    # Case 3: Bottle / drinking detection elevates risk to Medium and adds reason
    bottle_active = {"phone_detected": False, "bottle_detected": True, "bottle_confidence": 0.85, "bottle_duration_sec": 1.5, "is_bottle_alert": True}
    res_bottle = engine.evaluate(drowsy_safe, distract_safe, bottle_active)
    assert res_bottle["score"] > RISK_BAND_LOW_MAX
    assert any("Bottle" in r or "Drinking" in r for r in res_bottle["primary_reasons"])


def test_head_pose_variance():
    detector = DistractionDetector()
    assert len(detector.pose_history) == 0

    for _ in range(10):
        detector.pose_history.append((0.5, -0.2, 0.1))
    
    y_arr = [p[0] for p in detector.pose_history]
    p_arr = [p[1] for p in detector.pose_history]
    var_stable = float(np.var(y_arr) + np.var(p_arr))
    assert var_stable < 1.0

    for i in range(15):
        detector.pose_history.append((float(i * 10 - 70), float((i % 3) * 20 - 20), 0.0))
    y_arr_erratic = [p[0] for p in detector.pose_history]
    p_arr_erratic = [p[1] for p in detector.pose_history]
    var_erratic = float(np.var(y_arr_erratic) + np.var(p_arr_erratic))
    assert var_erratic > 50.0


def test_ml_risk_classifier():
    from core.risk_engine import MLRiskClassifier

    classifier = MLRiskClassifier()
    assert classifier.model is not None
    assert len(classifier.FEATURE_NAMES) == 6
    assert classifier.CLASSES == ["Low", "Medium", "High"]

    safe_features = {
        "blink_rate": 18.0,
        "perclos": 0.02,
        "yawn_frequency": 0.0,
        "head_pose_variance": 5.0,
        "phone_duration": 0.0,
        "stress_score": 0.10
    }
    pred_safe = classifier.predict(safe_features)
    assert "level" in pred_safe
    assert "probability" in pred_safe
    assert "probabilities" in pred_safe
    assert 0.0 <= pred_safe["probability"] <= 1.0
    assert pred_safe["level"] in ["Low", "Medium", "High"]

    danger_features = {
        "blink_rate": 4.0,
        "perclos": 0.35,
        "yawn_frequency": 4.0,
        "head_pose_variance": 120.0,
        "phone_duration": 5.0,
        "stress_score": 0.85
    }
    pred_danger = classifier.predict(danger_features)
    assert pred_danger["level"] in ["Medium", "High"]
    assert pred_danger["probabilities"]["High"] > pred_safe["probabilities"]["High"]


def test_risk_engine_hybrid_ml_integration():
    engine = RiskEngine()
    
    drowsy_safe = {"ear": 0.35, "blink_rate_bpm": 18, "is_eye_closed": False, "closure_duration_sec": 0.0, "perclos": 0.02, "is_yawning": False, "recent_yawns": 0}
    distract_safe = {"head_off_road": False, "gaze_off_road": False, "distraction_duration_sec": 0.0, "head_pose_variance": 3.0}
    phone_safe = {"phone_detected": False, "confidence": 0.0, "phone_duration_sec": 0.0}
    stretch_data = {"stress_score": 0.05, "dominant_emotion": "Neutral / Calm"}

    res = engine.evaluate(drowsy_safe, distract_safe, phone_safe, stretch_data)
    assert "ml_risk_probability" in res
    assert "ml_risk_level" in res
    assert res["level"] == "Low"
    assert res["ml_risk_level"] in ["Low", "Medium", "High"]
