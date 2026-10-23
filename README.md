# 🚗 GuardianDrive AI — Smart Driver Monitoring & Safety Co-Pilot

> **A real-time, privacy-first in-cabin driver safety system that runs 100% locally on standard CPU hardware.**  
> Powered by OpenCV, MediaPipe Face Mesh, Ultralytics YOLOv8, Scikit-Learn, and FastAPI.

---

## 🌟 Overview

Drowsy driving and smartphone distractions cause thousands of avoidable road accidents every year. Most commercial safety systems either require expensive cloud subscriptions with high latency, or make generic, ambiguous beeps that leave the driver wondering what went wrong.

**GuardianDrive AI** turns an ordinary webcam into an intelligent in-cabin co-pilot. It tracks facial landmarks, eye closure, yawning, 3D head rotation, iris gaze direction, handheld phone usage, and drinking while driving.

### Core Highlights
- **100% Local & CPU-Optimized**: Runs smoothly at ~30 FPS on standard laptops and desktops without requiring an NVIDIA GPU or cloud AI API keys.
- **Clear Spoken Voice Warnings**: No robotic beeps. It gives precise spoken feedback: *"Driver looking left. Face forward on the road"*, *"Put the phone away!"*, or *"Wake up! Eyes closed! Pull over safely, now."*
- **Smart Passenger Filtering**: Uses the driver's 468-point facial boundary to ignore phones or bottles held by passengers.
- **Smart Blink Suppression**: Distinguishes normal 200ms eye blinks from downward gaze, preventing false "looking down" alarms.
- **Hybrid Risk Engine**: Fuses established computer vision formulas (Eye Aspect Ratio, 3D solvePnP) with a 50-tree Scikit-Learn Random Forest classifier for explainable safety scores.
- **Emergency SOS & Safe Havens**: 1-click in-cabin SOS button speaks an emergency alert, emails live GPS coordinates with vehicle speed to safety contacts, and routes to nearby rest stops and hospitals via Google Maps.

---

## 🏗️ System Architecture & Data Flow

```
                          ┌──────────────────────────┐
                          │   Live Physical Webcam   │
                          │   (Direct 30 FPS Stream) │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │  MediaPipe Face Mesh     │
                          │  (478 Points with Irises)│
                          └─────────────┬────────────┘
                                        │
      ┌────────────────────────┬────────┴───────────────┬────────────────────────┐
      ▼                        ▼                        ▼                        ▼
┌──────────────┐      ┌─────────────────┐      ┌──────────────────┐     ┌────────────────┐
│  Drowsiness  │      │   Distraction   │      │ Object Detection │     │ Emotion/Stress │
│  - EAR (Eyes)│      │ - solvePnP 3D   │      │ - YOLOv8 Nano    │     │ - FACS Eyebrow │
│  - MAR (Yawn)│      │   Head Pose     │      │ - Phone (ID 67)  │     │   Compression  │
│  - PERCLOS   │      │ - Yaw/Pitch/Roll│      │ - Bottle (ID 39) │     │ - Mouth Frown/ │
│    (60s avg) │      │ - Iris Gaze     │      │ - Driver Face    │     │   Tension      │
│  - Microsleep│      │ - Smart Blink   │      │   Proximity Mask │     │ - Continuous   │
│    Alarm     │      │   Suppression   │      │   (Driver-Only)  │     │   Stress Index │
└──────┬───────┘      └────────┬────────┘      └────────┬─────────┘     └───────┬────────┘
       │                       │                        │                       │
       └───────────────────────┴────────┬───────────────┴───────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      Hybrid Risk Engine     │
                         │ ─────────────────────────── │
                         │ • 5-Factor Weighted Math    │
                         │ • Scikit-Learn Random Forest│
                         │ • Safety Override Circuits  │
                         │ • Fused Score (0.0 to 1.0)  │
                         └──────────────┬──────────────┘
                                        │
               ┌────────────────────────┴────────────────────────┐
               ▼                                                 ▼
┌─────────────────────────────┐                  ┌───────────────────────────────┐
│     Safety Interventions    │                  │    Dashboard & Telemetry      │
│ ─────────────────────────── │                  │ ───────────────────────────── │
│ • Offline Voice TTS Alerts  │                  │ • FastAPI MJPEG Video Stream  │
│ • Emergency In-Cabin SOS    │                  │ • Real-Time WebSockets HUD    │
│ • Dual-Recipient Email      │                  │ • Leaflet / Google Maps Route │
│   (Driver & Passenger)      │                  │ • Safe Haven 1-Click Routing  │
│ • Live Google Maps Nav Link │                  │ • SQLite DB (driver_logs.db)  │
│ • Pre-written Safety Advice │                  │ • Daily/Weekly Scorecards     │
└─────────────────────────────┘                  └───────────────────────────────┘
```

---

## 📁 Project Structure

```
AI-Driver Monitoring System/
├── core/
│   ├── camera.py             # Threaded video capture with auto-reconnect
│   ├── landmarks.py          # 478-point MediaPipe Face Mesh & iris extraction
│   ├── drowsiness.py         # EAR, MAR, rolling 60s PERCLOS & microsleep tracking
│   ├── distraction.py        # 3D head pose (solvePnP), iris gaze & blink suppression
│   ├── phone_detector.py     # YOLOv8n phone & bottle detection with passenger filter
│   ├── emotion.py            # FACS facial geometry analyzer & stress index
│   ├── risk_engine.py        # 5-factor mathematical model + Scikit-Learn Random Forest
│   ├── voice.py              # Offline pyttsx3 speech engine with per-alert cooldowns
│   ├── gps.py                # GPS tracker (device coordinates, speed, heading)
│   ├── maps_service.py       # Google Maps Platform service & Safe Haven finder
│   ├── notifier.py           # Multi-part RFC 2046 email notifier (plain text + HTML)
│   ├── advisory.py           # Safety recovery procedures & HTML sanitization
│   └── report_generator.py   # Daily, Weekly, and Monthly driving scorecards
├── database/
│   └── db_manager.py         # Thread-safe SQLite 3 database manager
├── data/
│   ├── driver_logs.db        # SQLite database (sessions, trips, telemetry, alerts)
│   └── risk_rf_model.pkl     # Pre-trained Random Forest risk classifier
├── models/
│   └── face_landmarker.task  # MediaPipe Face Landmarker model asset
├── ui/
│   ├── overlay.py            # OpenCV HUD renderer (mesh, 3D pose axes, bounding boxes)
│   ├── web_app.py            # FastAPI app, WebSocket telemetry, and REST endpoints
│   └── templates/
│       └── index.html        # Responsive web dashboard with interactive maps
├── tests/
│   ├── test_advanced_features.py  # Maps, emotions, reports, and GPS tests
│   ├── test_metrics.py            # EAR, MAR, head pose & ML risk tests
│   └── test_specific_alerts.py    # Voice phrases & distraction tests
├── .env.example              # Environment configuration template
├── config.py                 # Central configurations & calibrated thresholds
├── main.py                   # Main startup entrypoint (Web or Desktop mode)
├── run.py                    # Convenience launcher
├── requirements.txt          # Python dependencies
├── Dockerfile                # Debian-based container definition
├── docker-compose.yml        # Docker Compose service definition
└── yolov8n.pt                # YOLOv8 nano pre-trained weights
```

---

## 🔬 Core Capabilities

### 1. Drowsiness & Fatigue Detection (`core/drowsiness.py`)
- **Eye Aspect Ratio (EAR)**: Computes eye opening using 6 landmark points per eye. Values below `0.22` indicate eye closure. An Exponential Moving Average (EMA) filter smooths single-frame flicker.
- **Rolling 60-Second PERCLOS**: Measures the percentage of time eyes remain closed over the past 60 seconds. Values above `15%` indicate creeping fatigue before nodding off.
- **Yawn Detection (MAR)**: Monitors inner and outer vertical lip opening against mouth width. Requires sustained frames (`~2.0s`) and a debounce cooldown (`3.0s`) to avoid triggering on speech or laughter.
- **Microsleep Alarm**: Eyes remaining closed for more than `1.5 seconds` immediately sound a critical emergency alarm.

### 2. Head Pose & Directional Distraction (`core/distraction.py`)
- **3D Head Pose via `solvePnP`**: Solves Perspective-n-Point against a 3D facial model to compute Euler angles in degrees:
  - **Looking Left**: $\text{Yaw} < -22^\circ$
  - **Looking Right**: $\text{Yaw} > +22^\circ$
  - **Looking Down**: $\text{Pitch} < -18^\circ$ (lap, phone, or center console)
  - **Looking Up**: $\text{Pitch} > +18^\circ$
  - **Head Tilt / Slumping**: $|\text{Roll}| > 25^\circ$
- **Iris Gaze Tracking**: Evaluates inner/outer iris boundaries to spot off-road glances even when the head is facing forward.
- **Smart Blink Suppression**: Automatically suppresses downward gaze alerts while eyes are closed, eliminating false alarms from normal blinking.

### 3. Object Detection: Phone & Bottle (`core/phone_detector.py`)
- **YOLOv8 Nano**: Runs a lightweight CPU model (`yolov8n.pt`) with test-time augmentation and 448px inference for high recall on tilted devices.
- **Dual Detection**:
  - `cell phone` (COCO class ID 67): Flags calls, texting, and screen interaction.
  - `bottle` (COCO class ID 39): Detects drinking while driving, advising the driver to pull over safely first.
- **Passenger Filter**: Uses the driver's 468-point facial boundary to set an active interaction perimeter, ignoring devices in passenger seats.
- **Hold Timers**: Requires `1.0s` sustained detection before alarming to avoid transient false positives.

### 4. Emotion & Stress Recognition (`core/emotion.py`)
- **FACS Geometry**: Analyzes inner eyebrow compression (Corrugator supercilii contraction), eyebrow-to-eye spacing, and mouth curvature.
- **Adaptive Baseline**: Slowly adapts to the driver's resting neutral face geometry.
- **Continuous Stress Index**: Outputs a `0.0–1.0` index and classifies driver mood into `Calm`, `Stressed`, `Fatigued`, or `Frustrated`.

### 5. Hybrid Risk Assessment Engine (`core/risk_engine.py`)
- **5-Factor Mathematical Model**:
  $$\text{Risk}_{\text{math}} = w_{\text{ear}} f_{\text{ear}} + w_{\text{perclos}} f_{\text{perclos}} + w_{\text{yawn}} f_{\text{yawn}} + w_{\text{distract}} f_{\text{distract}} + w_{\text{phone}} f_{\text{phone}}$$
  *(Weights: EAR 0.20, PERCLOS 0.25, Yawn 0.15, Distraction 0.20, Phone/Bottle 0.20)*
- **Random Forest Classifier**: 50-tree Scikit-Learn ensemble trained on 6 behavioral signals (blink rate, PERCLOS, yawns, head pose variance, phone duration, stress). Falls back to a pure-NumPy ensemble if scikit-learn is absent.
- **Safety Overrides**: Active hazards (microsleep, active phone) immediately escalate the risk score to critical.
- **Risk Bands**:
  - 🟢 **Low Risk (0.00 – 0.35)**: Safe, attentive driving.
  - 🟡 **Medium Risk (0.35 – 0.70)**: Caution; mild fatigue or brief gaze drift.
  - 🔴 **High Risk (0.70 – 1.00)**: Critical danger; active intervention required.

### 6. Voice Alerts (`core/voice.py`)
- **100% Offline Speech Synthesis**: Uses native OS voices via `pyttsx3` (Windows SAPI5, Linux espeak, macOS NSSpeech).
- **Thread Silence Fix**: Initializes fresh engine instances per message and calls Windows COM initialization to prevent background thread lockups.
- **Independent Cooldowns**: 4-second timers per alert type prevent recent distraction warnings from suppressing urgent fatigue alarms.

| Alert | Spoken Warning |
| :--- | :--- |
| `DISTRACTION_LOOKING_LEFT` | *"Driver looking left. Face forward and focus on the road."* |
| `DISTRACTION_LOOKING_RIGHT` | *"Driver looking right. Keep your eyes forward on the road."* |
| `DISTRACTION_LOOKING_DOWN` | *"Look up! Do not look down while driving. Keep your eyes on the road."* |
| `PHONE_USAGE` | *"Put the phone away! Using a mobile phone while driving is dangerous."* |
| `BOTTLE_USAGE` | *"Do not drink while driving. Stop the vehicle safely before drinking."* |
| `MICROSLEEP_DROWSINESS` | *"Wake up! Eyes closed! Wake up immediately! Pull over safely, now."* |
| `YAWNING_FATIGUE` | *"Yawning detected. You are getting tired. Consider taking a rest break."* |
| `EMERGENCY_SOS` | *"Emergency SOS activated. Transmitting vehicle location and speed to safety contacts."* |

### 7. GPS, Google Maps & Safe Havens (`core/gps.py`, `core/maps_service.py`)
- **Authentic Telemetry**: Ingests genuine coordinates from browser Geolocation or GPS receivers. Filters out jitter under 3 meters so stationary vehicles report strictly $0.0\text{ km/h}$.
- **Route Breadcrumbs**: Maintains a rolling history of up to 200 waypoints.
- **Emergency Safe Havens**: Automatically finds nearby **Rest Areas**, **Fuel Stations**, **Cafes**, and **Hospitals** within 15 km with 1-click Google Maps navigation buttons.

### 8. Trip Registration & Dual Emergency SOS (`core/notifier.py`, `ui/web_app.py`)
- **Trip Registration**: Drivers or passengers can record their names, vehicle plate, and passenger emergency email before the trip.
- **In-Cabin SOS Button**: Speaks an audible confirmation and sends an emergency email to both the fleet manager and passenger emergency contact with exact coordinates, speed, and a direct Google Maps rescue link.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10 to 3.15**
- Standard USB or integrated webcam

### 1. Installation
Clone the repository, set up a virtual environment, and install dependencies:

```bash
# Clone the repository
git clone https://github.com/Hithesh279/AI-Driver-Monitoring-System.git
cd AI-Driver-Monitoring-System

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Linux users: ensure `libgl1`, `libglib2.0-0`, and `espeak` are installed via `apt-get`)*

### 2. Run the System

```bash
# Web Dashboard (recommended — auto-opens browser at http://localhost:8000)
python main.py

# Standalone OpenCV Desktop Window Mode (press 'q' to exit, 'c' to calibrate)
python main.py --desktop

# Headless / No auto-browser launch
python main.py --no-browser

# Docker Container
docker-compose up --build
```

---

## 🎯 Driver Baseline Calibration

Because every driver's seating height and facial geometry differ, you can calibrate the system in 2 seconds:

1. Sit in your natural, comfortable driving posture looking straight ahead.
2. Keep your face relaxed.
3. Trigger calibration:
   - **Web Mode**: Click **"Calibrate Baseline"** on the dashboard.
   - **Desktop Mode**: Press the **`c`** key in the video window.
4. The system locks your resting EAR, MAR, neutral head angle offsets, and facial relaxation.

---

## ⚙️ Configuration Reference (`config.py` / `.env`)

To customize settings, copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `CAMERA_INDEX` | `0` | Physical camera hardware index |
| `FRAME_WIDTH` / `HEIGHT` | `640` / `480` | Downscaled frame resolution for CPU speed |
| `EAR_THRESHOLD` | `0.22` | Eye closure threshold |
| `MICROSLEEP_TIME_SEC` | `1.5` | Continuous eye closure triggering microsleep alarm |
| `MAR_THRESHOLD` | `0.45` | Mouth Aspect Ratio threshold for yawn detection |
| `YAW_LIMIT_DEG` | `22.0°` | Head turned left/right threshold |
| `PITCH_LIMIT_DEG` | `18.0°` | Head tilted up/down threshold |
| `YOLO_CONF_THRESHOLD` | `0.25` | Confidence threshold for phone & bottle detection |
| `PHONE_PROXIMITY_RATIO` | `1.6` | Driver face proximity multiplier for passenger filtering |
| `DMS_VOICE_ENABLED` | `true` | Enables offline OS speech warnings |
| `DMS_EMAIL_ENABLED` | `false` | Enables SMTP emergency emails (`true` / `false`) |
| `DMS_SMTP_USER` | `""` | Sender email address |
| `DMS_SMTP_PASSWORD` | `""` | 16-character Gmail App Password |
| `DMS_EMAIL_ONLY_ON_SOS`| `true` | Restricts email dispatches strictly to SOS triggers |
| `GOOGLE_MAPS_API_KEY` | `""` | Optional Google Maps API key (Leaflet used if blank) |

---

## 📡 REST API & WebSockets Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web dashboard UI |
| `GET` | `/video_feed` | Low-latency MJPEG video stream with HUD annotations |
| `WS` | `/ws/telemetry` | Real-time WebSocket broadcasting live telemetry |
| `POST` | `/api/trip/start` | Registers driver, passenger, and emergency contact for the trip |
| `GET` | `/api/trip/current` | Returns active trip details |
| `POST` | `/api/emergency/sos` | In-cabin SOS button trigger (spoken alert + emergency email) |
| `POST` | `/api/test_alert` | Manually triggers a test alert cycle |
| `POST` | `/api/calibrate_driver` | Calibrates resting driver baseline |
| `GET` | `/api/session/history` | Fetches recent telemetry logs and incident alerts |
| `GET` | `/api/maps/config` | Google Maps configuration, status, and vehicle location |
| `GET` | `/api/maps/nearby_havens`| Finds closest safe havens (fuel, rest stops, hospitals) |
| `POST` | `/api/gps/update` | Streams authentic live coordinates from device |
| `GET` | `/api/maps/route` | Returns recorded GPS route breadcrumb trail |
| `GET` | `/api/reports/{timeframe}`| Returns driving scorecard JSON (`daily`, `weekly`, `monthly`) |
| `GET` | `/api/reports_view/html` | Printable executive HTML driving scorecard |

---

## 🗄️ Database & Logs (`data/driver_logs.db`)

All trip particulars, sessions, sampled telemetry, and incident alerts are stored locally in SQLite 3:
- **`trips`**: Trip IDs, driver/passenger info, vehicle license plates, status.
- **`sessions`**: Session duration, average and peak risk scores, total alerts, yawns, phone incidents.
- **`telemetry_logs`**: Periodic records of EAR, MAR, PERCLOS, Yaw, Pitch, Roll, and Risk scores.
- **`alerts`**: Recorded incidents with timestamps, severity levels, trigger values, and descriptions.

**Quick Inspection:**
```bash
python -c "import sqlite3; conn = sqlite3.connect('data/driver_logs.db'); print(conn.execute('SELECT alert_type, severity, details FROM alerts ORDER BY id DESC LIMIT 5').fetchall())"
```
Or open `http://localhost:8000/api/reports_view/html` in your browser for a formatted scorecard.

---

## 🧪 Testing

The repository includes a comprehensive 20-test automated test suite:

```bash
python -m pytest tests/ -v
```

- **`tests/test_advanced_features.py`**: Validates Haversine distance, Google Maps breadcrumbs, Safe Haven lookups, emotion recognition, report generation, stationary GPS zeroing, and emergency SOS email formatting.
- **`tests/test_metrics.py`**: Verifies EAR and MAR formulas, risk score bounding $[0.0, 1.0]$, head pose variance, and Random Forest risk predictions.
- **`tests/test_specific_alerts.py`**: Ensures all directional voice phrases exist and tests distraction classification logic.

---

## 🛠️ Real-World Engineering Insights

1. **Blinking vs Downward Gaze**: Natural blinks temporarily lower eyelid pitch. In `core/distraction.py`, a gaze-suppression circuit pauses downward gaze alerts while eyes are closed, eliminating false alarms.
2. **Passenger Phone Filter**: In `core/phone_detector.py`, an active interaction perimeter is calculated from the driver's 468-point face boundary. Any phone detected outside this perimeter is ignored.
3. **Windows pyttsx3 Thread Deadlock**: On Windows, calling `pyttsx3` repeatedly from background threads causes COM deadlocks. In `core/voice.py`, we initialize `pythoncom.CoInitialize()`, create a fresh engine instance per message, and cleanly dispose of it.
4. **Parked Vehicle GPS Drift**: GPS sensors jitter by 1–5 meters even when parked. In `core/maps_service.py`, shifts under 3.0 meters are filtered out, locking stationary speed strictly to $0.0\text{ km/h}$.

---

## 📄 License & Acknowledgments

This project is licensed under the **MIT License**.

- **MediaPipe**: Cross-platform machine learning for perceptual computing.
- **Ultralytics YOLOv8**: Real-time object detection models.
- **OpenCV**: Computer vision library.
- **FastAPI**: Modern web framework for high-performance APIs.
- **Leaflet & OpenStreetMap**: Universal interactive mapping.

---

<div align="center">
  <b>Built with ❤️ for driver safety and accident prevention. Drive safe!</b>
</div>
