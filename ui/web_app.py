"""
FastAPI Web Application & Live Telemetry Server.
Provides MJPEG video streaming, WebSocket real-time telemetry, and REST endpoints
for the core Driver Monitoring pipeline (drowsiness, distraction, phone usage, risk).
"""

import atexit
from contextlib import asynccontextmanager
import cv2
import time
import uuid
import json
import asyncio
import threading
import numpy as np
from pathlib import Path
from typing import Set, Dict, Any, Optional, List, Tuple
from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import SERVER_HOST, SERVER_PORT, JPEG_QUALITY, EMAIL_ONLY_ON_SOS
from database.db_manager import DatabaseManager
from core.camera import CameraStream
from core.landmarks import FaceLandmarkDetector
from core.drowsiness import DrowsinessDetector
from core.distraction import DistractionDetector
from core.phone_detector import PhoneDetector
from core.risk_engine import RiskEngine
from core.advisory import get_advisory, format_email_body, format_email_text
from core.gps import GPSTracker
from core.voice import VoiceAlerter, ALERT_PHRASES
from core.notifier import EmailNotifier
from core.emotion import EmotionDetector
from core.maps_service import GoogleMapsService
from core.report_generator import ReportGenerator
from ui.overlay import draw_hud_overlay

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


class DriverSystemPipeline:
    """
    Owns the camera, the core detectors, emotion recognition, Google Maps, and the risk engine.
    Runs a background thread that continuously:
      camera frame -> face landmarks -> drowsiness + distraction + phone + emotion
                    -> risk engine -> HUD overlay -> JPEG + telemetry
    """

    def __init__(self):
        self.session_id = f"trip_{uuid.uuid4().hex[:8]}"
        self.db = DatabaseManager()
        self.db.start_session(self.session_id)

        self.camera = CameraStream()
        self.camera.start()

        self.landmarks_detector = FaceLandmarkDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.distraction_detector = DistractionDetector()
        self.phone_detector = PhoneDetector()
        self.emotion_detector = EmotionDetector()
        self.risk_engine = RiskEngine()
        self.gps = GPSTracker()
        self.maps_service = GoogleMapsService(self.gps)
        self.report_generator = ReportGenerator(self.db)
        self.voice = VoiceAlerter()
        self.email_notifier = EmailNotifier()

        self.latest_jpeg = None
        self.latest_landmarks: Optional[List[Tuple[int, int]]] = None
        self.latest_telemetry: Dict[str, Any] = {}
        self.active_websockets: Set[WebSocket] = set()
        self.lock = threading.Lock()
        self.is_running = True
        self._worker_running = True
        self._session_ended = False

        self.last_db_log_time = 0.0
        self.active_alert_start_times: Dict[str, float] = {}

        # Trip & Occupant Particulars
        self.current_trip = {
            "trip_id": f"TRIP-{int(time.time())}",
            "driver_name": "",
            "driver_email": "",
            "passenger_name": "",
            "passenger_emergency_email": "",
            "vehicle_id": "",
            "start_time": time.time(),
            "status": "NOT_REGISTERED"
        }

        self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker_thread.start()

    def stop_background_worker(self):
        """Stops the background web processing loop so desktop mode can use detectors without race conditions."""
        self._worker_running = False
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)

    def stop(self):
        """Gracefully ends the active database session and releases hardware resources."""
        self.is_running = False
        self._worker_running = False
        if not self._session_ended:
            self._session_ended = True
            try:
                self.db.end_session(self.session_id)
            except Exception:
                pass
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            self.landmarks_detector.close()
        except Exception:
            pass

    def _processing_loop(self):
        fps_tracker = []
        last_time = time.time()

        while self.is_running and self._worker_running:
            try:
                frame = self.camera.get_frame()
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

                self._process_frame(frame, avg_fps, current_time)
                time.sleep(0.005)
            except Exception:
                time.sleep(0.02)

    def _process_frame(self, frame: np.ndarray, avg_fps: float, current_time: float) -> np.ndarray:
        # 1. Extract facial landmarks (MediaPipe Face Mesh)
        landmarks = self.landmarks_detector.process_frame(frame)
        self.latest_landmarks = landmarks

        # 2. Drowsiness (EAR / MAR / PERCLOS) + Distraction (head pose / gaze)
        if landmarks is not None:
            drowsiness_data = self.drowsiness_detector.update(landmarks)
            distraction_data = self.distraction_detector.update(landmarks, frame.shape)
        else:
            drowsiness_data = {
                "ear": 0.0, "mar": 0.0, "perclos": 0.0, "is_eye_closed": False,
                "closure_duration_sec": 0.0, "is_drowsy": False, "is_microsleep": False,
                "is_yawning": False, "total_yawns": self.drowsiness_detector.total_yawns,
                "recent_yawns": 0, "blink_rate_bpm": 0
            }
            distraction_data = {
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "gaze_direction": "Unknown",
                "head_off_road": False, "gaze_off_road": False,
                "distraction_duration_sec": 0.0, "is_distracted": False,
                "rvec": None, "tvec": None
            }

        # 3. Emotion & Psychological Stress Recognition (PPT Module 03)
        emotion_data = self.emotion_detector.update(landmarks, drowsiness_data)

        # 4. YOLOv8 phone-in-hand detection with driver interaction zone filtering
        phone_data = self.phone_detector.detect(frame, landmarks)

        # 5. Fused 5-factor + Random Forest risk evaluation with emotion & stress signals
        risk_data = self.risk_engine.evaluate(drowsiness_data, distraction_data, phone_data, emotion_data)

        # 6. GPS Positioning & Google Maps Platform tracking
        gps_pos = self.gps.get_location()
        map_telemetry = self.maps_service.update_position(gps_pos["lat"], gps_pos["lon"])

        # 7. Draw the HUD overlay for the live video feed
        rendered_frame = draw_hud_overlay(
            frame, landmarks, drowsiness_data, distraction_data, phone_data, risk_data, fps=avg_fps
        )

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        _, buffer = cv2.imencode('.jpg', rendered_frame, encode_param)
        with self.lock:
            self.latest_jpeg = buffer.tobytes()

        telemetry = {
            "session_id": self.session_id,
            "timestamp": current_time,
            "fps": round(avg_fps, 1),
            "ear": drowsiness_data.get("ear", 0.0),
            "mar": drowsiness_data.get("mar", 0.0),
            "perclos": drowsiness_data.get("perclos", 0.0),
            "blink_rate_bpm": drowsiness_data.get("blink_rate_bpm", 0),
            "is_yawning": drowsiness_data.get("is_yawning", False),
            "total_yawns": drowsiness_data.get("total_yawns", 0),
            "is_drowsy": drowsiness_data.get("is_drowsy", False),
            "is_microsleep": drowsiness_data.get("is_microsleep", False),
            "yaw": distraction_data.get("yaw", 0.0),
            "pitch": distraction_data.get("pitch", 0.0),
            "roll": distraction_data.get("roll", 0.0),
            "gaze_direction": distraction_data.get("gaze_direction", "Forward"),
            "is_distracted": distraction_data.get("is_distracted", False),
            "phone_detected": phone_data.get("phone_detected", False),
            "phone_duration": phone_data.get("phone_duration_sec", 0.0),
            "bottle_detected": phone_data.get("bottle_detected", False),
            "bottle_duration": phone_data.get("bottle_duration_sec", 0.0),
            "dominant_emotion": emotion_data.get("dominant_emotion", "Calm"),
            "stress_score": emotion_data.get("stress_score", 0.0),
            "is_stressed": emotion_data.get("is_stressed", False),
            "emotion_probabilities": emotion_data.get("probabilities", {}),
            "risk_score": risk_data.get("score", 0.0),
            "math_score": risk_data.get("math_score", risk_data.get("score", 0.0)),
            "risk_level": risk_data.get("level", "Low"),
            "status_text": risk_data.get("status_text", "Normal"),
            "primary_reasons": risk_data.get("primary_reasons", []),
            "ml_risk": risk_data.get("ml_risk", {}),
            "gps": gps_pos,
            "speed_kmh": map_telemetry.get("speed_kmh", 0.0),
            "is_stationary": map_telemetry.get("is_stationary", True),
            "heading_deg": map_telemetry.get("heading_deg", 0.0),
            "breadcrumbs": self.maps_service.get_breadcrumbs(),
            "nearby_havens": self.maps_service.find_nearby_safe_havens(gps_pos["lat"], gps_pos["lon"]),
            "trip": self.current_trip,
        }
        self.latest_telemetry = telemetry

        # Periodic database logging (every ~1s)
        if (current_time - self.last_db_log_time) >= 1.0:
            self.db.log_telemetry(
                self.session_id, current_time,
                telemetry["ear"], telemetry["mar"], telemetry["perclos"],
                telemetry["yaw"], telemetry["pitch"], telemetry["roll"],
                telemetry["phone_detected"], telemetry["risk_score"], telemetry["risk_level"],
                total_yawns=telemetry["total_yawns"]
            )
            self.last_db_log_time = current_time

        # ------------------------------------------------------------------
        # Active Alerts: evaluates granular mistakes and distractions
        # Dispatches specific voice intervention and emergency alert with
        # vehicle location, speed, and exact incident reason.
        # ------------------------------------------------------------------
        active_alerts = []

        # 1. Critical Eye Closure & Fatigue Alerts
        if drowsiness_data.get("is_microsleep", False) or drowsiness_data.get("closure_duration_sec", 0.0) >= 1.2:
            active_alerts.append("MICROSLEEP_DROWSINESS")
        elif drowsiness_data.get("is_drowsy", False) and drowsiness_data.get("closure_duration_sec", 0.0) >= 0.8:
            active_alerts.append("DROWSY_DROOPING_EYES")
        elif drowsiness_data.get("is_yawning", False):
            active_alerts.append("YAWNING_FATIGUE")

        # 2. Cell Phone & Bottle In-Cabin Usage
        if phone_data.get("is_phone_alert", False) or (phone_data.get("phone_detected", False) and phone_data.get("phone_duration_sec", 0.0) >= 0.8):
            active_alerts.append("PHONE_USAGE")
        if phone_data.get("is_bottle_alert", False):
            active_alerts.append("BOTTLE_USAGE")

        # 3. Granular Directional & Postural Distraction
        if distraction_data.get("is_distracted", False):
            dtype = distraction_data.get("distraction_type", "")
            pitch = distraction_data.get("pitch", 0.0)
            yaw = distraction_data.get("yaw", 0.0)
            roll = distraction_data.get("roll", 0.0)
            gaze_dir = distraction_data.get("gaze_direction", "Forward")

            if dtype == "LOOKING_DOWN" or pitch < -16.0 or gaze_dir == "Looking Down":
                active_alerts.append("DISTRACTION_LOOKING_DOWN")
            elif dtype == "LOOKING_LEFT" or yaw < -20.0 or (gaze_dir == "Left" and yaw < -8.0):
                active_alerts.append("DISTRACTION_LOOKING_LEFT")
            elif dtype == "LOOKING_RIGHT" or yaw > 20.0 or (gaze_dir == "Right" and yaw > 8.0):
                active_alerts.append("DISTRACTION_LOOKING_RIGHT")
            elif dtype == "LOOKING_UP" or pitch > 18.0:
                active_alerts.append("DISTRACTION_LOOKING_UP")
            elif dtype == "HEAD_TILT" or abs(roll) > 24.0:
                active_alerts.append("DISTRACTION_HEAD_TILT")
            elif dtype == "GAZE_OFF_ROAD" or distraction_data.get("gaze_off_road", False):
                active_alerts.append("DISTRACTION_GAZE")
            else:
                active_alerts.append("OFF_ROAD_DISTRACTION")

        # 4. Psychological Stress
        if emotion_data.get("is_stressed", False) and emotion_data.get("stress_score", 0.0) >= 0.80:
            active_alerts.append("HIGH_STRESS")

        alert_reason_map = {
            "MICROSLEEP_DROWSINESS": "Driver Asleep at Wheel / Microsleep Eye Closure",
            "DROWSY_DROOPING_EYES": "Driver Eyelids Drooping / High Drowsiness",
            "YAWNING_FATIGUE": "Driver Yawning / Excessive Fatigue",
            "PHONE_USAGE": "Using Mobile Phone — Keep your eyes on the road",
            "BOTTLE_USAGE": "Driver Drinking / Bottle Near Face While In Motion",
            "DISTRACTION_LOOKING_DOWN": "Driver Looking Down at Lap or Console",
            "DISTRACTION_LOOKING_LEFT": "Driver Looking Left Away From Road",
            "DISTRACTION_LOOKING_RIGHT": "Driver Looking Right Away From Road",
            "DISTRACTION_LOOKING_UP": "Driver Head Tilted Up Away From Road",
            "DISTRACTION_HEAD_TILT": "Driver Head Tilted Sideways / Slumping",
            "DISTRACTION_GAZE": "Driver Gaze Diverted Away From Road",
            "OFF_ROAD_DISTRACTION": "Driver Distracted / Extended Gaze Diverted Away from Road",
            "HIGH_STRESS": "High Driver Stress and Agitation Detected",
            "EMERGENCY_SOS": "Emergency In-Cabin SOS Distress Button Triggered",
        }

        # Priority hierarchy for voice alert so the driver gets the most critical specific instruction first
        VOICE_PRIORITY_ORDER = [
            "EMERGENCY_SOS",
            "MICROSLEEP_DROWSINESS",
            "PHONE_USAGE",
            "DISTRACTION_LOOKING_DOWN",
            "DISTRACTION_LOOKING_LEFT",
            "DISTRACTION_LOOKING_RIGHT",
            "DISTRACTION_LOOKING_UP",
            "DISTRACTION_HEAD_TILT",
            "DISTRACTION_GAZE",
            "OFF_ROAD_DISTRACTION",
            "BOTTLE_USAGE",
            "DROWSY_DROOPING_EYES",
            "YAWNING_FATIGUE",
            "HIGH_STRESS"
        ]

        primary_voice_key = None
        for p_key in VOICE_PRIORITY_ORDER:
            if p_key in active_alerts:
                primary_voice_key = p_key
                break

        if primary_voice_key:
            voice_phrase = ALERT_PHRASES.get(primary_voice_key, "Warning. Please stay alert.")
            self.voice.speak(voice_phrase, key=primary_voice_key)

        advisories = []
        for alert_type in active_alerts:
            advisory = get_advisory(alert_type, telemetry["risk_level"])
            specific_reason = alert_reason_map.get(alert_type, advisory["condition"])

            # Log a fresh DB row only the moment this alert type starts
            if alert_type not in self.active_alert_start_times:
                self.active_alert_start_times[alert_type] = current_time
                reasons_str = ", ".join(telemetry["primary_reasons"])
                details_str = f"{specific_reason} ({reasons_str})" if reasons_str and specific_reason not in reasons_str else specific_reason
                self.db.log_alert(
                    self.session_id, current_time, alert_type, advisory.get("severity", "HIGH"),
                    telemetry["risk_score"], details=details_str
                )

            speed_val = float(telemetry.get("speed_kmh", 0.0))
            gps_val = telemetry.get("gps", {"lat": 0.0, "lon": 0.0})

            # Automated email dispatch is disabled when EMAIL_ONLY_ON_SOS is True (strictly reserved for SOS button)
            if not EMAIL_ONLY_ON_SOS:
                subject_prefix = "🚨 [CRITICAL ALERT]" if advisory["severity"] == "CRITICAL" else "⚠️ [SAFETY ALERT]"
                email_subject = (
                    f"{subject_prefix} {specific_reason} — "
                    f"Speed: {speed_val:.1f} km/h, Location: Lat {gps_val['lat']:.4f}, Lon {gps_val['lon']:.4f}"
                )

                # Route to passenger emergency contact as primary recipient (taken input from user)
                recipients = []
                user_passenger_email = (self.current_trip.get("passenger_emergency_email") or "").strip()
                if user_passenger_email and "@" in user_passenger_email:
                    recipients.append(user_passenger_email)
                else:
                    for dr in self.email_notifier.default_recipients:
                        if dr not in recipients:
                            recipients.append(dr)

                self.email_notifier.send_alert(
                    subject=email_subject,
                    html_body=format_email_body(
                        advisory, telemetry, gps_val,
                        reason=specific_reason,
                        trip_details=self.current_trip
                    ),
                    text_body=format_email_text(
                        advisory, telemetry, gps_val,
                        reason=specific_reason,
                        trip_details=self.current_trip
                    ),
                    key=alert_type,
                    recipients=recipients
                )
            advisories.append(advisory)

        # Clear the start-time for any alert type that is no longer active
        for alert_type in list(self.active_alert_start_times.keys()):
            if alert_type not in active_alerts:
                del self.active_alert_start_times[alert_type]

        telemetry["advisories"] = advisories
        telemetry["advisory"] = advisories[0] if advisories else None
        return rendered_frame


# Pipeline singleton, started on import
pipeline = DriverSystemPipeline()
atexit.register(pipeline.stop)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    yield
    pipeline.stop()


app = FastAPI(title="Driver Monitoring System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Driver Monitoring System</h2>")


async def generate_frames():
    try:
        while pipeline.is_running:
            if pipeline.latest_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + pipeline.latest_jpeg + b'\r\n')
            await asyncio.sleep(0.033)
    except (asyncio.CancelledError, GeneratorExit):
        pass


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    pipeline.active_websockets.add(websocket)
    try:
        while pipeline.is_running:
            if pipeline.latest_telemetry:
                await websocket.send_text(json.dumps(pipeline.latest_telemetry))
            await asyncio.sleep(0.1)  # 10 Hz telemetry push
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        pass
    finally:
        pipeline.active_websockets.discard(websocket)


@app.post("/api/calibrate")
async def api_calibrate():
    """Resets the head-pose baseline to the driver's current 'looking forward' position."""
    yaw = pipeline.latest_telemetry.get("yaw", 0.0)
    pitch = pipeline.latest_telemetry.get("pitch", 0.0)
    roll = pipeline.latest_telemetry.get("roll", 0.0)
    pipeline.distraction_detector.calibrate(yaw, pitch, roll)
    return {"status": "success", "message": "Head pose calibrated to center"}


@app.post("/api/simulation_mode")
async def api_simulation_mode(mode: str = "live"):
    """Legacy endpoint notice: System operates exclusively on authentic live hardware sensor telemetry."""
    return {
        "status": "info",
        "message": "Simulation modes are disabled. System operates exclusively on authentic live hardware sensor telemetry.",
        "mode": "live_hardware"
    }


@app.get("/api/risk/ml")
async def api_risk_ml():
    """Returns the latest Random Forest classifier output: predicted class, confidence, feature importances."""
    ml_data = pipeline.latest_telemetry.get("ml_risk", {})
    return {"status": "success", "model": "Random Forest Classifier (scikit-learn)", "ml_risk": ml_data}


@app.get("/api/gps")
async def api_gps():
    return {"status": "success", "location": pipeline.latest_telemetry.get("gps", pipeline.gps.get_location())}


class TripRegistrationRequest(BaseModel):
    driver_name: str
    driver_email: Optional[str] = ""
    passenger_name: str
    passenger_emergency_email: str
    vehicle_id: Optional[str] = "AP-04-REG-01"


@app.post("/api/trip/start")
async def api_trip_start(req: TripRegistrationRequest):
    """
    Registers Trip Particulars before commencing the journey.
    Sets passenger emergency email as primary safety alert recipient.
    """
    trip_id = f"TRIP-{int(time.time())}"
    pipeline.current_trip = {
        "trip_id": trip_id,
        "driver_name": req.driver_name.strip() or "Registered Driver",
        "driver_email": (req.driver_email or "").strip(),
        "passenger_name": req.passenger_name.strip() or "Registered Passenger",
        "passenger_emergency_email": req.passenger_emergency_email.strip(),
        "vehicle_id": (req.vehicle_id or "").strip() or "AP-04-REG-01",
        "start_time": time.time(),
        "status": "ACTIVE"
    }

    # Save trip to persistent database
    pipeline.db.start_trip(
        trip_id=trip_id,
        session_id=pipeline.session_id,
        driver_name=pipeline.current_trip["driver_name"],
        driver_email=pipeline.current_trip["driver_email"],
        passenger_name=pipeline.current_trip["passenger_name"],
        passenger_emergency_email=pipeline.current_trip["passenger_emergency_email"],
        vehicle_id=pipeline.current_trip["vehicle_id"]
    )

    return {
        "status": "success",
        "message": f"Trip registered successfully for passenger {pipeline.current_trip['passenger_name']}. Emergency alerts will be dispatched to {pipeline.current_trip['passenger_emergency_email']}.",
        "trip": pipeline.current_trip
    }


@app.get("/api/trip/current")
async def api_trip_current():
    """Returns currently active trip and occupant particulars."""
    return {"status": "success", "trip": pipeline.current_trip}


@app.post("/api/test_alert")
async def api_test_alert(alert_type: str = "PHONE_USAGE"):
    """Manually fires one alert cycle (voice + email) for verification purposes."""
    telemetry = pipeline.latest_telemetry
    gps_loc = pipeline.latest_telemetry.get("gps", pipeline.gps.get_location())
    speed = float(telemetry.get("speed_kmh", 0.0))
    advisory = get_advisory(alert_type, telemetry.get("risk_level", "Medium"))
    reason_text = f"SAFETY ALERT VERIFICATION: {advisory['condition']}"

    pipeline.voice.speak(ALERT_PHRASES.get(alert_type, "Warning. Please stay alert."), key=alert_type, cooldown=0)

    subject_prefix = "🚨 [TEST CRITICAL]" if advisory["severity"] == "CRITICAL" else "⚠️ [TEST ALERT]"
    email_subject = (
        f"{subject_prefix} {advisory['condition']} — "
        f"Speed: {speed:.1f} km/h, Location: Lat {gps_loc['lat']:.4f}, Lon {gps_loc['lon']:.4f}"
    )

    # Emails are strictly sent ONLY when the SOS distress button is clicked (unless EMAIL_ONLY_ON_SOS=false)
    sent = False
    recipients = []
    if not EMAIL_ONLY_ON_SOS:
        user_passenger_email = (pipeline.current_trip.get("passenger_emergency_email") or "").strip()
        if user_passenger_email and "@" in user_passenger_email:
            recipients.append(user_passenger_email)
        else:
            for dr in pipeline.email_notifier.default_recipients:
                if dr not in recipients:
                    recipients.append(dr)

        sent = pipeline.email_notifier.send_alert(
            subject=email_subject,
            html_body=format_email_body(
                advisory, telemetry, gps_loc,
                reason=reason_text,
                trip_details=pipeline.current_trip
            ),
            text_body=format_email_text(
                advisory, telemetry, gps_loc,
                reason=reason_text,
                trip_details=pipeline.current_trip
            ),
            key=alert_type, cooldown=0,
            recipients=recipients
        )
    return {
        "status": "success",
        "advisory": advisory,
        "email_sent": sent,
        "email_enabled": pipeline.email_notifier.enabled,
        "speed_kmh": speed,
        "location": gps_loc,
        "reason": reason_text,
        "recipients": recipients
    }


@app.post("/api/emergency/sos")
@app.post("/api/sos")
async def api_emergency_sos():
    """
    Immediate In-Cabin Emergency SOS Trigger.
    Instantly transmits vehicle location, live speed, and distress reason
    to passenger emergency contact email, and issues immediate audible feedback.
    """
    telemetry = pipeline.latest_telemetry
    gps_pos = telemetry.get("gps", pipeline.gps.get_location())
    speed = float(telemetry.get("speed_kmh", 0.0))
    advisory = get_advisory("EMERGENCY_SOS", "Critical")
    reason_text = "EMERGENCY SOS: In-cabin distress button manually triggered by driver"

    # Immediate voice announcement
    pipeline.voice.speak(
        ALERT_PHRASES.get("EMERGENCY_SOS", "Emergency SOS activated. Transmitting vehicle location and speed to safety contacts."),
        key="EMERGENCY_SOS",
        cooldown=0
    )

    # Immediate email dispatch to passenger emergency contact
    recipients = []
    if pipeline.current_trip.get("passenger_emergency_email"):
        recipients.append(pipeline.current_trip["passenger_emergency_email"])
    for dr in pipeline.email_notifier.default_recipients:
        if dr not in recipients:
            recipients.append(dr)

    subject = (
        f"🚨 [CRITICAL SOS] Emergency Distress Button Activated — "
        f"Speed: {speed:.1f} km/h, Location: Lat {gps_pos['lat']:.4f}, Lon {gps_pos['lon']:.4f}"
    )
    email_body = format_email_body(
        advisory=advisory,
        telemetry=telemetry,
        gps=gps_pos,
        reason=reason_text,
        trip_details=pipeline.current_trip
    )
    email_text = format_email_text(
        advisory=advisory,
        telemetry=telemetry,
        gps=gps_pos,
        reason=reason_text,
        trip_details=pipeline.current_trip
    )

    sent = pipeline.email_notifier.send_alert(
        subject=subject,
        html_body=email_body,
        text_body=email_text,
        key="EMERGENCY_SOS",
        cooldown=0,
        recipients=recipients
    )

    # Log to incident database
    pipeline.db.log_alert(
        pipeline.session_id,
        time.time(),
        "EMERGENCY_SOS",
        "CRITICAL",
        1.0,
        details=f"Manual SOS Triggered. Speed: {speed:.1f} km/h. Location: {gps_pos['lat']:.5f}, {gps_pos['lon']:.5f}"
    )

    return {
        "status": "success",
        "message": "Emergency SOS alert dispatched immediately to passenger emergency safety contact.",
        "email_sent": sent,
        "email_enabled": pipeline.email_notifier.enabled,
        "gps": gps_pos,
        "speed_kmh": speed,
        "reason": reason_text,
        "recipients": recipients
    }


@app.get("/api/session/history")
@app.get("/api/history")
async def api_session_history():
    logs = pipeline.db.get_recent_telemetry(pipeline.session_id, limit=60)
    alerts = pipeline.db.get_recent_alerts(pipeline.session_id, limit=20)
    return {"session_id": pipeline.session_id, "telemetry": logs, "alerts": alerts}


# ----------------------------------------------------------------------
# Google Maps Platform Integration Endpoints
# ----------------------------------------------------------------------
@app.get("/api/maps/config")
async def api_maps_config():
    """Returns Google Maps configuration, API key status, vehicle position, and route breadcrumbs."""
    return pipeline.maps_service.get_google_maps_config()


@app.get("/api/maps/nearby_havens")
async def api_maps_nearby_havens():
    """Returns nearest safe havens (rest areas, fuel stations, cafes, hospitals) with Google Maps navigation links."""
    loc = pipeline.latest_telemetry.get("gps", pipeline.gps.get_location())
    havens = pipeline.maps_service.find_nearby_safe_havens(loc["lat"], loc["lon"])
    return {
        "status": "success",
        "current_location": loc,
        "speed_kmh": pipeline.latest_telemetry.get("speed_kmh", 0.0),
        "havens": havens,
    }


@app.post("/api/gps/update")
async def api_gps_update(payload: Dict[str, Any]):
    """Receives authentic live GPS telemetry streamed directly from the client device."""
    try:
        lat = float(payload.get("lat"))
        lon = float(payload.get("lon"))
        speed_kmh = float(payload.get("speed_kmh", 0.0)) if payload.get("speed_kmh") is not None else 0.0
        heading = float(payload.get("heading_deg", 0.0)) if payload.get("heading_deg") is not None else 0.0

        pipeline.gps.update_live_location(lat, lon, speed_kmh=speed_kmh, heading_deg=heading)
        map_status = pipeline.maps_service.update_position(
            lat, lon, reported_speed_kmh=speed_kmh, reported_heading=heading
        )
        return {
            "status": "success",
            "lat": lat,
            "lon": lon,
            "speed_kmh": map_status["speed_kmh"],
            "is_stationary": map_status["is_stationary"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/maps/route")
async def api_maps_route():
    """Returns the GPS breadcrumb trail history for mapping the vehicle trajectory."""
    return {
        "status": "success",
        "breadcrumbs": pipeline.maps_service.get_breadcrumbs(),
        "count": len(pipeline.maps_service.breadcrumbs)
    }


# ----------------------------------------------------------------------
# Automated Driving Performance Reports (PPT Module 05)
# ----------------------------------------------------------------------
@app.get("/api/reports/{timeframe}")
async def api_driving_report(timeframe: str = "daily"):
    """Generates a structured driving report (daily, weekly, or monthly) with safety scores and coaching tips."""
    if timeframe not in ["daily", "weekly", "monthly"]:
        timeframe = "daily"
    report = pipeline.report_generator.generate_report(timeframe)
    return {"status": "success", "report": report}


@app.get("/api/reports_view/html", response_class=HTMLResponse)
async def api_reports_view_html(timeframe: str = "daily"):
    """Renders a printable HTML driving summary report."""
    html_content = pipeline.report_generator.render_html_report(timeframe)
    return HTMLResponse(content=html_content)


# ----------------------------------------------------------------------
# Full Driver Baseline Calibration (High Accuracy)
# ----------------------------------------------------------------------
@app.post("/api/calibrate_driver")
async def api_calibrate_driver():
    """
    Performs comprehensive driver baseline calibration:
    Locks resting EAR, MAR, head orientation center, and facial relaxation.
    """
    raw_ear = pipeline.latest_telemetry.get("ear", 0.28)
    raw_mar = pipeline.latest_telemetry.get("mar", 0.20)
    yaw = pipeline.latest_telemetry.get("yaw", 0.0)
    pitch = pipeline.latest_telemetry.get("pitch", 0.0)
    roll = pipeline.latest_telemetry.get("roll", 0.0)

    pipeline.drowsiness_detector.calibrate(baseline_ear=raw_ear, baseline_mar=raw_mar)
    pipeline.distraction_detector.calibrate(yaw, pitch, roll)
    if pipeline.latest_landmarks is not None:
        pipeline.emotion_detector.calibrate(pipeline.latest_landmarks)

    return {
        "status": "success",
        "message": "Driver baseline calibrated successfully",
        "calibrated_values": {
            "baseline_ear": round(raw_ear, 3),
            "baseline_mar": round(raw_mar, 3),
            "head_pose_zeroed": True,
            "emotion_baseline_set": pipeline.latest_landmarks is not None
        }
    }


def run_server():
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    run_server()
