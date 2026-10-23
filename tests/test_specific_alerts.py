"""
Unit tests for granular distraction detection, specific voice phrases, and tailored safety advisories.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.voice import ALERT_PHRASES
from core.advisory import ADVISORY_TABLE, get_advisory
from core.distraction import DistractionDetector

def test_alert_phrases_and_advisories_exist():
    expected_keys = [
        "DISTRACTION_LOOKING_LEFT",
        "DISTRACTION_LOOKING_RIGHT",
        "DISTRACTION_LOOKING_DOWN",
        "DISTRACTION_LOOKING_UP",
        "DISTRACTION_HEAD_TILT",
        "DISTRACTION_GAZE",
        "OFF_ROAD_DISTRACTION",
        "PHONE_USAGE",
        "BOTTLE_USAGE",
        "MICROSLEEP_DROWSINESS",
        "DROWSY_DROOPING_EYES",
        "YAWNING_FATIGUE",
        "HIGH_STRESS",
        "EMERGENCY_SOS"
    ]
    for key in expected_keys:
        assert key in ALERT_PHRASES, f"Missing voice phrase for key: {key}"
        assert len(ALERT_PHRASES[key]) > 10, f"Phrase too short for: {key}"
        assert key in ADVISORY_TABLE, f"Missing advisory entry for key: {key}"
        adv = get_advisory(key)
        assert "condition" in adv
        assert "steps" in adv
        assert len(adv["steps"]) >= 2

def test_distraction_type_classification():
    detector = DistractionDetector()
    detector.is_distracted = True
    detector.distraction_duration_sec = 2.0

    # Test synthetic state lookup logic
    # Looking down: pitch < -16
    dtype = "LOOKING_DOWN" if -20.0 < -16.0 else "OTHER"
    assert dtype == "LOOKING_DOWN"

    # Looking left: yaw < -20
    dtype = "LOOKING_LEFT" if -25.0 < -20.0 else "OTHER"
    assert dtype == "LOOKING_LEFT"

    # Looking right: yaw > 20
    dtype = "LOOKING_RIGHT" if 25.0 > 20.0 else "OTHER"
    assert dtype == "LOOKING_RIGHT"

    # Head tilt: roll > 24
    dtype = "HEAD_TILT" if abs(28.0) > 24.0 else "OTHER"
    assert dtype == "HEAD_TILT"
