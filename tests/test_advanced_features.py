"""
Unit tests for Google Maps Platform Integration, Emotion & Stress Recognition,
Automated Reports Generator, Database Isolation, and High-Accuracy Calibration.
"""

import sys
import tempfile
from pathlib import Path

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

from core.maps_service import GoogleMapsService, haversine_distance_km
from core.emotion import EmotionDetector
from core.report_generator import ReportGenerator
from core.drowsiness import DrowsinessDetector
from core.distraction import DistractionDetector
from database.db_manager import DatabaseManager


def test_haversine_and_google_maps_service():
    maps = GoogleMapsService()
    # Test distance Kadapa (14.4673, 78.8242) to nearby point
    d = haversine_distance_km(14.4673, 78.8242, 14.4773, 78.8342)
    assert 1.0 < d < 2.5

    # Test position update & breadcrumb recording
    status = maps.update_position(14.4673, 78.8242)
    assert status["lat"] == 14.4673
    assert status["lon"] == 78.8242
    assert "speed_kmh" in status
    assert "heading_deg" in status
    assert len(maps.get_breadcrumbs()) >= 1

    # Test Nearby Safe Havens finder with Google Maps deep-links
    havens = maps.find_nearby_safe_havens(14.4673, 78.8242)
    assert len(havens) >= 3
    for h in havens:
        assert "name" in h
        assert "category" in h
        assert "distance_km" in h
        assert "google_maps_nav_url" in h
        assert "google_maps_view_url" in h
        assert "google.com/maps" in h["google_maps_nav_url"]


def test_emotion_and_stress_detector():
    detector = EmotionDetector()
    assert detector.smoothed_stress == pytest.approx(0.10, abs=0.05)

    # Synthetic neutral face landmarks (478 points)
    # Eyes at (100, 100) and (200, 100) -> scale = 100
    landmarks = [(0, 0) for _ in range(500)]
    landmarks[33] = (100, 100)
    landmarks[263] = (200, 100)
    landmarks[107] = (135, 80)
    landmarks[336] = (165, 80)
    landmarks[66] = (120, 75)
    landmarks[296] = (180, 75)
    landmarks[159] = (120, 95)
    landmarks[386] = (180, 95)
    landmarks[61] = (125, 140)
    landmarks[291] = (175, 140)
    landmarks[0] = (150, 132)
    landmarks[17] = (150, 148)

    # Update for several frames
    for _ in range(4):
        res = detector.update(landmarks)

    assert "dominant_emotion" in res
    assert res["dominant_emotion"] in ["Calm", "Stressed", "Fatigued", "Frustrated"]
    assert "stress_score" in res
    assert 0.0 <= res["stress_score"] <= 1.0
    assert "probabilities" in res
    assert len(res["probabilities"]) == 4


def test_report_generator_timeframes(tmp_path=None):
    if tmp_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)
    db = DatabaseManager(db_path=tmp_path / "report_test.db")
    import time
    now = time.time()
    db.start_session("sess_rep_1", start_time=now - 120)
    db.log_telemetry("sess_rep_1", now - 60, 0.28, 0.18, 0.04, 2.0, -1.0, 0.0, False, 0.18, "Low", total_yawns=1)
    db.log_alert("sess_rep_1", now - 50, "DISTRACTION_LOOKING_LEFT", "HIGH", 0.55, "Looking left")
    db.log_alert("sess_rep_1", now - 40, "DISTRACTION_LOOKING_DOWN", "HIGH", 0.62, "Looking down")
    db.log_alert("sess_rep_1", now - 30, "DROWSY_DROOPING_EYES", "HIGH", 0.72, "Drooping eyes")

    gen = ReportGenerator(db)

    # Test Daily report and verify granular alerts roll up into OFF_ROAD_DISTRACTION and FATIGUE_WARNING
    daily = gen.generate_report("daily")
    assert daily["timeframe"] == "Daily"
    assert "safety_score" in daily
    assert 0 <= daily["safety_score"] <= 100
    assert daily["alert_breakdown"]["OFF_ROAD_DISTRACTION"] == 2
    assert daily["alert_breakdown"]["FATIGUE_WARNING"] == 1
    assert "grade" in daily
    assert "recommendations" in daily
    assert len(daily["recommendations"]) >= 1

    # Test HTML report rendering
    html = gen.render_html_report("daily")
    assert "<!DOCTYPE html>" in html
    assert "GuardianDrive AI" in html
    assert "Driver Safety Score" in html


def test_high_accuracy_calibration_and_smoothing():
    # Test Drowsiness calibration
    drowsy_det = DrowsinessDetector()
    drowsy_det.calibrate(baseline_ear=0.32, baseline_mar=0.18)
    assert drowsy_det.baseline_ear == pytest.approx(0.32, abs=0.001)
    assert drowsy_det.baseline_mar == pytest.approx(0.18, abs=0.001)
    assert drowsy_det.is_calibrated is True

    # Test Distraction head pose smoothing & reset on calibrate
    dist_det = DistractionDetector()
    dist_det.smoothed_yaw = 15.0
    dist_det.is_distracted = True
    dist_det.calibrate(current_yaw=2.0, current_pitch=-1.0, current_roll=0.5)
    assert dist_det.baseline_yaw == pytest.approx(2.0, abs=0.001)
    assert dist_det.baseline_pitch == pytest.approx(-1.0, abs=0.001)
    assert dist_det.smoothed_yaw == 0.0
    assert dist_det.is_distracted is False


def test_live_speed_and_stationary_zero_telemetry():
    from core.gps import GPSTracker
    gps = GPSTracker()
    maps = GoogleMapsService(gps_tracker=gps)

    # 1. Initial state must be stationary at 0.0 km/h
    assert maps.current_speed_kmh == 0.0
    assert maps.is_stationary is True

    # 2. Update with stationary coordinates (desk/parking: jitter < 3 meters)
    lat1, lon1 = 14.467300, 78.824200
    st1 = maps.update_position(lat1, lon1)
    assert st1["speed_kmh"] == 0.0
    assert st1["is_stationary"] is True

    # Sub-meter GPS satellite jitter (0.00001 deg is ~1.1 meters)
    lat2, lon2 = 14.467308, 78.824205
    st2 = maps.update_position(lat2, lon2)
    assert st2["speed_kmh"] == 0.0
    assert st2["is_stationary"] is True

    # 3. Direct device GPS speed (e.g. 62.4 km/h while on highway)
    st3 = maps.update_position(14.470000, 78.830000, reported_speed_kmh=62.4, reported_heading=45.0)
    assert st3["speed_kmh"] == 62.4
    assert st3["is_stationary"] is False
    assert st3["heading_deg"] == 45.0

    # 4. Stopping the vehicle
    st4 = maps.update_position(14.470000, 78.830000, reported_speed_kmh=0.0)
    assert st4["speed_kmh"] == 0.0
    assert st4["is_stationary"] is True

    # 5. GPS tracker never applies random walk drift
    loc1 = gps.get_location()
    loc2 = gps.get_location()
    assert loc1["lat"] == loc2["lat"]
    assert loc1["lon"] == loc2["lon"]


def test_immediate_email_formatting_and_sos():
    from core.advisory import get_advisory, format_email_body
    from core.voice import ALERT_PHRASES

    # 1. Verify SOS advisory and voice phrase
    sos_adv = get_advisory("EMERGENCY_SOS", "Critical")
    assert sos_adv["severity"] == "CRITICAL"
    assert "EMERGENCY_SOS" in ALERT_PHRASES

    # 2. Verify format_email_body includes exact Location, Speed, and Reason
    telemetry = {
        "speed_kmh": 72.5,
        "ear": 0.11,
        "perclos": 0.35,
        "phone_detected": True,
        "is_distracted": False,
        "yaw": 5.0,
        "pitch": -3.0,
        "dominant_emotion": "Fatigued",
        "stress_score": 0.75,
        "risk_score": 0.88,
        "session_id": "test_session_alert"
    }
    gps = {"lat": 14.467312, "lon": 78.824215}
    reason = "Driver Asleep at Wheel / Microsleep Eye Closure"

    adv = get_advisory("MICROSLEEP_DROWSINESS", "Critical")
    body = format_email_body(adv, telemetry, gps, reason=reason)

    # Check reason is displayed
    assert reason in body
    # Check speed is displayed
    assert "72.5 km/h" in body
    # Check coordinates and Google Maps navigation deep-link
    assert "14.467312" in body
    assert "78.824215" in body
    assert "https://www.google.com/maps/dir/?api=1" in body
    assert "destination=14.467312,78.824215" in body


def test_trip_registration_and_db_persistence(tmp_path=None):
    if tmp_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)
    # Test DB trip persistence in isolated test database
    db_file = tmp_path / "test_dms.db"
    db = DatabaseManager(db_path=str(db_file))
    assert Path(db.db_path) == db_file.resolve()

    trip_id = "TRIP-TEST-9988"
    db.start_trip(
        trip_id=trip_id,
        session_id="session_123",
        driver_name="Ramesh Kumar",
        driver_email="ramesh@transport.com",
        passenger_name="Hemanth Kumar",
        passenger_emergency_email="family.safety@example.com",
        vehicle_id="AP-04-TX-9988"
    )

    trip = db.get_trip(trip_id)
    assert trip is not None
    assert trip["driver_name"] == "Ramesh Kumar"
    assert trip["driver_email"] == "ramesh@transport.com"
    assert trip["passenger_name"] == "Hemanth Kumar"
    assert trip["passenger_emergency_email"] == "family.safety@example.com"
    assert trip["vehicle_id"] == "AP-04-TX-9988"
    assert trip["status"] == "ACTIVE"

    latest = db.get_latest_trip()
    assert latest is not None
    assert latest["trip_id"] == trip_id


def test_traditional_email_layout_and_passenger_emergency_routing():
    from core.advisory import get_advisory, format_email_body, format_email_text
    from core.notifier import EmailNotifier

    trip_details = {
        "trip_id": "TRIP-2026-X88",
        "driver_name": "Ramesh Kumar",
        "driver_email": "ramesh@transport.com",
        "passenger_name": "Hemanth Kumar",
        "passenger_emergency_email": "passenger.emergency@family.com",
        "vehicle_id": "AP-04-TX-9988"
    }

    telemetry = {
        "speed_kmh": 65.4,
        "ear": 0.12,
        "perclos": 0.40,
        "phone_detected": False,
        "dominant_emotion": "Fatigued",
        "stress_score": 0.65,
        "risk_score": 0.85,
        "session_id": "trip_session_01"
    }
    gps = {"lat": 14.467350, "lon": 78.824280}
    reason = "Driver Asleep at Wheel / Microsleep Eye Closure"

    adv = get_advisory("MICROSLEEP_DROWSINESS", "Critical")
    html_email = format_email_body(adv, telemetry, gps, reason=reason, trip_details=trip_details)
    text_email = format_email_text(adv, telemetry, gps, reason=reason, trip_details=trip_details)

    # 1. Verify Traditional Formal Dispatch Title and Headers (HTML)
    assert "GUARDIANDRIVE AI TELEMATICS DISPATCH" in html_email
    assert "URGENT SAFETY NOTICE" in html_email
    assert "TO:</b> Designated Passenger Emergency Contact (<b>passenger.emergency@family.com</b>)" in html_email
    assert "DMS-EMERGENCY-" in html_email

    # 2. Verify Table 1 Occupant Particulars (HTML)
    assert "1. Trip &amp; Occupant Particulars" in html_email
    assert "Hemanth Kumar" in html_email
    assert "passenger.emergency@family.com" in html_email
    assert "Ramesh Kumar" in html_email
    assert "AP-04-TX-9988" in html_email

    # 3. Verify Table 2 Incident Telemetry & Live Speed (HTML)
    assert "2. Incident Telemetry &amp; Vehicle Status" in html_email
    assert "Driver Asleep at Wheel / Microsleep Eye Closure" in html_email
    assert "65.4 km/h (In Motion)" in html_email

    # 4. Verify Section 3 Navigation & Directives (HTML)
    assert "3. Live Vehicle Location &amp; Google Maps Navigation" in html_email
    assert "Open Google Maps Turn-by-Turn Directions to Vehicle" in html_email
    assert "destination=14.467350,78.824280" in html_email
    assert "4. Required Actions for Designated Emergency Contact" in html_email

    # 5. Verify Plain-Text Version (Text Msg)
    assert "GUARDIANDRIVE AI TELEMATICS DISPATCH" in text_email
    assert "TO: Designated Passenger Emergency Contact (passenger.emergency@family.com)" in text_email
    assert "Passenger Name:              Hemanth Kumar" in text_email
    assert "Recorded Vehicle Speed:      65.4 km/h (In Motion)" in text_email
    assert "Google Maps Turn-by-Turn:" in text_email
    assert "destination=14.467350,78.824280" in text_email

    # 6. Verify dynamic recipient passing with dual text & html in EmailNotifier (stubbing _send so tests don't make live SMTP calls)
    notifier = EmailNotifier()
    notifier._send = lambda *args, **kwargs: None
    dispatched = notifier.send_alert(
        subject="Emergency Alert",
        html_body=html_email,
        text_body=text_email,
        recipients=["passenger.emergency@family.com"]
    )
    # Returns True if SMTP credentials enabled, False if disabled
    assert dispatched == notifier.enabled


def test_safe_havens_non_blocking_performance():
    import time
    maps = GoogleMapsService()
    t0 = time.time()
    havens = maps.find_nearby_safe_havens(14.4673, 78.8242)
    duration_sec = time.time() - t0
    # Must complete almost instantaneously (< 50ms) without stalling the video frame thread
    assert duration_sec < 0.05
    assert len(havens) >= 3


def test_gaze_suppression_on_closed_eyes():
    dist_det = DistractionDetector()
    landmarks = [(0, 0) for _ in range(500)]
    # Set closed eye landmarks where vertical aperture is tiny (< 3px)
    landmarks[33] = (100, 100)
    landmarks[133] = (130, 100)
    landmarks[468] = (115, 101)
    landmarks[159] = (115, 100)
    landmarks[145] = (115, 101)  # height = 1px (closed eye)

    landmarks[362] = (170, 100)
    landmarks[263] = (200, 100)
    landmarks[473] = (185, 101)
    landmarks[386] = (185, 100)
    landmarks[374] = (185, 101)  # height = 1px (closed eye)

    direction, deviation = dist_det.estimate_gaze_direction(landmarks)
    # Must be suppressed to Forward/0.0 deviation when eyelids are shut
    assert direction == "Forward"
    assert deviation == 0.0


def test_html_escaping_in_advisory_and_report(tmp_path=None):
    from core.advisory import get_advisory, format_email_body
    if tmp_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)
    db = DatabaseManager(db_path=tmp_path / "escape_test.db")
    gen = ReportGenerator(db)

    malicious_trip = {
        "trip_id": "TRIP-XSS-101",
        "driver_name": "<script>alert('pwned')</script>",
        "driver_email": "bad@evil.com",
        "passenger_name": "<b>Injected</b>",
        "passenger_emergency_email": "safe@contact.com",
        "vehicle_id": "<img src=x onerror=alert(1)>"
    }
    telemetry = {"speed_kmh": 0.0, "risk_score": 0.5}
    gps = {"lat": 14.4673, "lon": 78.8242}
    adv = get_advisory("MICROSLEEP_DROWSINESS", "Critical")

    import html
    email = format_email_body(adv, telemetry, gps, reason="<marquee>Danger</marquee>", trip_details=malicious_trip)
    assert "<script>" not in email
    assert html.escape(malicious_trip["driver_name"]) in email
    assert "<img src=x" not in email
    assert html.escape(malicious_trip["vehicle_id"]) in email
    assert "<marquee>" not in email
    assert html.escape("<marquee>Danger</marquee>") in email

    report = gen.render_html_report("daily")
    assert "<script>" not in report


def test_email_only_on_sos_button():
    from config import EMAIL_ONLY_ON_SOS
    assert EMAIL_ONLY_ON_SOS is True
