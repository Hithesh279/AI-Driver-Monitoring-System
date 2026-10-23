# 🚗 GuardianDrive AI — Smart Driver Monitoring & Safety Co-Pilot

> **A real-time, privacy-first in-cabin driver safety system that runs 100% locally on standard CPU hardware.**  
> Powered by OpenCV, MediaPipe Face Mesh, Ultralytics YOLOv8, Scikit-Learn, and FastAPI.

---

## 🌟 Welcome to GuardianDrive AI

Every year, thousands of tragic road accidents occur because a driver fell asleep at the wheel for just a second, or took their eyes off the road to glance down at a smartphone. Most commercial warning systems either cost a fortune, require dedicated automotive cloud subscriptions with high latency, or make generic, annoying beeps that leave the driver wondering: *"Wait, what did I do wrong?"*

**GuardianDrive AI** was engineered from the ground up to solve those exact real-world problems. It turns any ordinary webcam or USB camera into an intelligent, attentive co-pilot. It watches over eye fatigue, yawning, 3D head rotation, iris gaze direction, handheld phone usage, and drinking while driving. 

When your attention drifts or exhaustion sets in, GuardianDrive AI speaks up with clear, calm, natural human voice alerts—telling you *exactly* what happened. In genuine emergencies, an in-cabin SOS button triggers an instant dispatch containing your exact vehicle coordinates, speed, and 1-click Google Maps rescue navigation directly to your family or fleet safety manager.

---

## 💡 Why We Built This (And What Makes It Different)

We designed GuardianDrive AI with five strict engineering principles:

1. **100% Local & CPU-Optimized (Zero Cloud Latency)**: You do not need an expensive NVIDIA GPU or a paid cloud AI subscription. The entire pipeline—from 478-point facial landmarking to YOLO object detection and Random Forest risk classification—runs smoothly on an everyday laptop CPU at ~30 FPS.
2. **Specific Spoken Voice Warnings (No Robot Beeps)**: An alarm beep is ambiguous. GuardianDrive AI uses offline OS speech synthesis to deliver clear spoken commands: *"Driver looking left. Face forward on the road"*, *"Put the phone away!"*, or *"Wake up! Eyes closed! Pull over safely, now."*
3. **No False Alarms From Passengers**: If a passenger sitting in the passenger seat picks up their phone, GuardianDrive AI checks the 468-point driver face boundary and ignores the passenger device entirely.
4. **No False Alarms From Normal Blinking**: Closing your eyes for 200 milliseconds to blink naturally lowers your eye pitch. Conventional algorithms mistake this for "looking down at your lap". We built a smart blink-suppression filter so you never get nagged for blinking.
5. **Hybrid AI Engine (Math + Machine Learning)**: We refuse to treat safety as an unexplainable "black box". We combine established geometric formulas (Eye Aspect Ratio, 3D solvePnP) with a trained 50-tree Scikit-Learn Random Forest classifier. You get both rock-solid mathematical explainability and nuanced pattern recognition.

---

## 🏗️ System Architecture & Data Pipeline

Here is how data flows from your physical camera lens to the dashboard, speakers, and emergency contacts:

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
│    (60s avg) │      │ - Iris Gaze     │      │ - Face Bounding  │     │   Tension      │
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
│ • Offline Voice TTS Alerts  │                  │ • FastAPI MJPEG Stream        │
│ • Emergency In-Cabin SOS    │                  │ • Real-Time WebSockets HUD    │
│ • Dual-Recipient Email      │                  │ • Leaflet / Google Maps Route │
│   (Driver & Passenger)      │                  │ • Safe Haven 1-Click Routing  │
│ • Live Google Maps Nav Link │                  │ • SQLite DB (driver_logs.db)  │
│ • Pre-written Safety Advice │                  │ • Daily/Weekly Scorecards     │
└─────────────────────────────┘                  └───────────────────────────────┘
```

---

## 🔬 Under The Hood: The Core Modules

GuardianDrive AI is divided into modular, self-contained components located in the `core/`, `database/`, and `ui/` directories.

```
AI-Driver Monitoring System/
├── core/
│   ├── camera.py             # Direct hardware video streaming thread with auto-reconnect
│   ├── landmarks.py          # 478-point MediaPipe Face Mesh & iris coordinate extractor
│   ├── drowsiness.py         # EAR, MAR, rolling 60s PERCLOS, and microsleep detector
│   ├── distraction.py        # 3D head pose (solvePnP), iris gaze & blink suppression
│   ├── phone_detector.py     # YOLOv8n phone & bottle detector with driver proximity filter
│   ├── emotion.py            # FACS facial geometry analyzer & continuous stress index
│   ├── risk_engine.py        # 5-factor mathematical model + Scikit-Learn Random Forest
│   ├── voice.py              # Offline pyttsx3 speech engine with per-alert cooldowns
│   ├── gps.py                # Authentic GPS tracker (device coordinates, speed, heading)
│   ├── maps_service.py       # Google Maps Platform service & Emergency Safe Haven finder
│   ├── notifier.py           # Multi-part RFC 2046 email notifier (plain text + HTML)
│   ├── advisory.py           # Deterministic safety recovery procedures & HTML sanitization
│   └── report_generator.py   # Daily, Weekly, and Monthly driving performance scorecards
├── database/
│   └── db_manager.py         # Thread-safe SQLite 3 database manager with re-entrant locks
├── data/
│   ├── driver_logs.db        # SQLite database (sessions, trips, telemetry_logs, alerts)
│   └── risk_rf_model.pkl     # Pre-trained Scikit-Learn Random Forest risk classifier
├── models/
│   └── face_landmarker.task  # MediaPipe Face Landmarker model asset
├── ui/
│   ├── overlay.py            # OpenCV HUD renderer (mesh, 3D pose axes, bounding boxes)
│   ├── web_app.py            # FastAPI application, WebSocket server, REST endpoints
│   └── templates/
│       └── index.html        # Responsive dark-theme dashboard with interactive maps
├── tests/
│   ├── test_advanced_features.py  # Maps, emotions, reports, and GPS tests
│   ├── test_metrics.py            # Mathematical bounds, EAR, MAR & ML tests
│   └── test_specific_alerts.py    # Voice phrases & directional distraction tests
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

### 1. Camera & Video Pipeline (`core/camera.py`)
- **Direct Hardware Access**: Streams video directly from your webcam, USB capture card, or automotive telematics camera using OpenCV's `VideoCapture`.
- **Fault-Tolerant Auto-Reconnect**: If a USB webcam is temporarily bumped or unplugged, a background reconnect loop automatically attempts recovery across camera indices (`0` and `1`) without crashing the application.
- **Branded Standby Canvas**: While searching for a camera, it renders an informative dark standby screen indicating hardware connection status rather than an ugly black box or fatal error.

### 2. MediaPipe Landmark Extractor (`core/landmarks.py`)
- **478 Facial Anchors**: Extracts 468 3D mesh points across the face plus 10 high-precision iris landmarks (`468-477`).
- **Dual-API Compatibility**: Automatically detects whether your environment uses legacy `mp.solutions.face_mesh` or modern `mediapipe.tasks.python.vision.FaceLandmarker`.
- **Automatic Model Provisioning**: If `models/face_landmarker.task` is missing, it automatically downloads the lightweight 16-bit float model from Google's official storage bucket on first run.

### 3. Drowsiness, Fatigue & Microsleep (`core/drowsiness.py`)
- **Eye Aspect Ratio (EAR)**: Computes eye opening using 6 key landmark points per eye:
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
  Values below `0.22` indicate eye closure. An Exponential Moving Average (EMA) filter smooths out single-frame measurement flicker.
- **Mouth Aspect Ratio (MAR) & Yawn Verification**: Measures inner and outer vertical lip opening relative to mouth width. It requires sustained frames (`~2.0s` at 30 FPS) combined with a cooldown period (`3.0s`) so talking, smiling, or singing does not falsely register as a yawn.
- **Rolling 60-Second PERCLOS**: Calculates the percentage of time eyes remain closed over the past 60 seconds:
  $$\text{PERCLOS} = \frac{\text{Frames with Closed Eyes in Last 60s}}{\text{Total Frames in Last 60s}}$$
  Values above `15%` indicate significant physiological exhaustion. This detects creeping drowsiness long before a driver's head actually drops.
- **Microsleep Alarm**: If eyes remain closed for more than `1.5 seconds`, a critical emergency alarm sounds immediately.
- **Blink Counter & Rate**: Tracks individual blinks and calculates live Blink Rate per Minute (BPM).

### 4. Head Pose & Directional Distraction (`core/distraction.py`)
- **3D Head Pose via `solvePnP`**: Maps 6 facial landmarks (nose tip, chin, eye corners, mouth corners) against a generic 3D human head model. OpenCV's `solvePnP` computes camera rotation matrices decomposed into exact Euler angles in degrees:
  - **Looking Left**: $\text{Yaw} < -22^\circ$ (checks left mirror or side window for too long).
  - **Looking Right**: $\text{Yaw} > +22^\circ$ (looking at passenger seat or out right window).
  - **Looking Down**: $\text{Pitch} < -18^\circ$ (looking at lap, center console, or texting).
  - **Looking Up**: $\text{Pitch} > +18^\circ$ (gazing at sun visor or ceiling).
  - **Head Slump / Tilt**: $|\text{Roll}| > 25^\circ$ (driver slumping sideways due to sleepiness).
- **Iris Gaze Tracking**: Evaluates the ratio between inner and outer iris boundaries to detect visual wander even when the head remains pointed forward.
- **Smart Blink Suppression**: Blinking naturally tilts the calculated eye pitch downward. The algorithm suppresses gaze alerts while eyes are closed, eliminating false "looking down" alerts.
- **Head Pose Variance**: Maintains a 10-second sliding window of head angular variance to spot erratic or restless head movements.

### 5. Object Detection: Phone & Bottle (`core/phone_detector.py`)
- **Ultralytics YOLOv8 Nano**: Runs a compact, CPU-friendly model (`yolov8n.pt`) tuned for maximum recall:
  - Higher inference resolution (`448px`) catches small or rotated devices.
  - Lowered confidence threshold (`0.25`) detects phones held at steep angles against the ear.
  - Test-Time Augmentation (`augment=True`) improves recall on tilted objects.
- **Dual-Class Detection**:
  - `cell phone` (COCO class ID 67): Flags phone calls, texting, and screen interaction.
  - `bottle` (COCO class ID 39): Detects drinking while driving, advising the driver to stop safely first.
- **Driver Proximity Filter**: Extracts the driver's full 468-point face bounding box and defines an active interaction perimeter. Any phone detected outside this boundary (such as in a passenger's hand) is completely ignored!
- **Hold Timers**: Sustained detection of `1.0s` is required before triggering an alert, preventing false alarms from brief reflections or hand movements.

### 6. Emotion & Stress Recognition (`core/emotion.py`)
- **Facial Action Coding System (FACS)**: Analyzes subtle muscular contractions across the face:
  - *Corrugator supercilii contraction*: Measured by the normalized distance between inner eyebrows (`107` and `336`). A tightening brow indicates stress or anxiety.
  - *Eyebrow elevation*: Distance between eyebrow arches and eye tops.
  - *Mouth tension & curvature*: Evaluates lip corner elevation relative to center lip anchor to distinguish relaxed neutral expressions from tight grimaces or frowns.
- **Adaptive Baseline**: Slowly adapts to each individual driver's resting neutral face geometry (`0.5%` weight per frame).
- **Continuous Output**: Produces a `0.0–1.0` continuous Stress Index and classifies emotional state into `Calm`, `Stressed`, `Fatigued`, or `Frustrated`.

### 7. Hybrid Risk Assessment Engine (`core/risk_engine.py`)
- **5-Factor Mathematical Model**: Computes a normalized hazard index across five transparent factors:
  $$\text{Risk}_{\text{math}} = w_{\text{ear}} f_{\text{ear}} + w_{\text{perclos}} f_{\text{perclos}} + w_{\text{yawn}} f_{\text{yawn}} + w_{\text{distract}} f_{\text{distract}} + w_{\text{phone}} f_{\text{phone}}$$
  *Default Weights*: $\text{EAR}=0.20$, $\text{PERCLOS}=0.25$, $\text{Yawn}=0.15$, $\text{Distraction}=0.20$, $\text{Phone}=0.20$.
- **Scikit-Learn Random Forest Classifier**: A 50-tree ensemble trained on 6 behavioral signals:
  1. `blink_rate` (blinks per minute)
  2. `perclos` (rolling 60s eye closure ratio)
  3. `yawn_frequency` (active yawns + recent count)
  4. `head_pose_variance` (10s angular variance)
  5. `phone_duration` (continuous seconds held)
  6. `stress_score` (FACS stress index)
- **Pure-NumPy Fallback**: If Scikit-Learn is not installed in the active environment, an in-memory NumPy probabilistic forest takes over automatically.
- **Safety Override Circuits**: Critical hazards trigger instant escalation:
  - Microsleep $\rightarrow$ Immediate Risk $\ge 0.85$.
  - Sustained eye closure $\rightarrow$ Immediate Risk $\ge 0.75$.
  - Phone active $\rightarrow$ Immediate Risk $\ge 0.75$.
  - Distraction + Phone simultaneous $\rightarrow$ Immediate Risk $\ge 0.90$.
- **Risk Bands**:
  - 🟢 **Low Risk (0.00 – 0.35)**: Safe, attentive driving.
  - 🟡 **Medium Risk (0.35 – 0.70)**: Caution; mild fatigue or brief gaze drift.
  - 🔴 **High Risk (0.70 – 1.00)**: Critical danger; active intervention required.

### 8. Voice Alerts & Speech Engine (`core/voice.py`)
- **100% Offline Speech Synthesis**: Uses `pyttsx3` to talk directly through your laptop or vehicle speakers using native OS voices (Windows SAPI5, Linux espeak, macOS NSSpeech).
- **The "Thread Silence" Bug Fix**: `pyttsx3` has a notorious bug when called repeatedly from background threads (it speaks once and then permanently goes mute). GuardianDrive AI fixes this by initializing a fresh engine instance per utterance and explicitly calling `pythoncom.CoInitialize()` on Windows.
- **Independent Per-Alert Cooldowns**: Each mistake has an independent 4-second timer. A recent phone warning will never block or delay an urgent fatigue alert!

| Alert Identifier | Natural Voice Warning Spoken to Driver |
| :--- | :--- |
| `DISTRACTION_LOOKING_LEFT` | *"Driver looking left. Face forward and focus on the road."* |
| `DISTRACTION_LOOKING_RIGHT` | *"Driver looking right. Keep your eyes forward on the road."* |
| `DISTRACTION_LOOKING_DOWN` | *"Look up! Do not look down while driving. Keep your eyes on the road."* |
| `DISTRACTION_LOOKING_UP` | *"Eyes on the road. Keep your head level and watch ahead."* |
| `DISTRACTION_HEAD_TILT` | *"Head tilt detected. Please sit upright and focus on driving."* |
| `DISTRACTION_GAZE` | *"Eyes off the road. Focus your attention forward."* |
| `PHONE_USAGE` | *"Put the phone away! Using a mobile phone while driving is dangerous."* |
| `BOTTLE_USAGE` | *"Do not drink while driving. Stop the vehicle safely before drinking."* |
| `MICROSLEEP_DROWSINESS` | *"Wake up! Eyes closed! Wake up immediately! Pull over safely, now."* |
| `DROWSY_DROOPING_EYES` | *"Warning. Your eyes are drooping. Stay alert!"* |
| `YAWNING_FATIGUE` | *"Yawning detected. You are getting tired. Consider taking a rest break."* |
| `FATIGUE_WARNING` | *"High fatigue detected. Please plan a rest stop soon."* |
| `HIGH_STRESS` | *"High stress detected. Take a deep breath and stay calm."* |
| `EMERGENCY_SOS` | *"Emergency SOS activated. Transmitting vehicle location and speed to safety contacts."* |

### 9. GPS & Google Maps Platform Integration (`core/gps.py`, `core/maps_service.py`)
- **Authentic Vehicle Coordinates**: Receives authentic coordinates streamed directly from the client browser Geolocation API or an onboard hardware GPS receiver (NMEA serial).
- **No Artificial Drift**: Unlike naive demos that add random walks to coordinates, GuardianDrive AI records genuine points. When stationary, GPS speed is strictly $0.0\text{ km/h}$ and position jitter under 3 meters is filtered out.
- **Haversine Distance**: Calculates exact distances between coordinates in kilometers.
- **Breadcrumb Trail**: Stores a rolling history of up to 200 route waypoints to draw the driven path on the live map.
- **Safe Haven Finder**: When fatigue or elevated risk is detected, the system searches within a 15 km radius for nearby:
  - ⛽ **Fuel Stations**
  - 🛑 **Highway Rest Areas**
  - ☕ **Cafes & Coffee Stops**
  - 🏥 **Hospitals & Emergency Medical Centers**
  Each safe haven card includes direct 1-click **"Navigate via Google Maps"** links.

### 10. Trip Registration & Dual Emergency SOS (`core/notifier.py`, `core/advisory.py`)
- **Trip Registration**: Before departing, drivers or passengers can register the driver name, passenger name, vehicle license plate, and the passenger's family emergency email.
- **In-Cabin SOS Button**: When clicked, the system:
  1. Speaks an emergency confirmation aloud over the speakers.
  2. Dispatches an immediate high-priority email to both the driver's safety manager and the passenger's family contact.
  3. Includes exact GPS coordinates, current speed, vital telemetry (EAR, fatigue level, risk score), and a direct Google Maps link to locate the vehicle.
- **Multi-Part RFC 2046 Email Format**: Contains both a clean plain-text fallback and a responsive HTML layout with safety steps.
- **XSS Sanitization**: All user-provided fields are sanitized with `html.escape` to ensure security.

### 11. Automated Driving Performance Scorecards (`core/report_generator.py`)
- Generates structured driving summaries across three timeframes:
  - **Daily**: Past 24 hours
  - **Weekly**: Past 7 days
  - **Monthly**: Past 30 days
- **Overall Driver Safety Score (0–100)**: Evaluates driving hours, incident counts, average risk, and PERCLOS to assign letter grades:
  - `A+` (95–100) | `A` (85–94) | `B` (70–84) | `C` (55–69) | `D` (<55)
- **Printable HTML Scorecards**: Access `http://localhost:8000/api/reports_view/html` to generate an executive report with personalized coaching tips.

### 12. Thread-Safe SQLite Database (`database/db_manager.py`)
- Stores all data locally in `data/driver_logs.db`.
- **Thread-Local Connections**: Avoids cross-thread SQLite sharing issues by using thread-local storage combined with a re-entrant lock (`threading.RLock`).
- **Cross-Platform Journal Mode**: Uses `PRAGMA journal_mode=DELETE;` ensuring reliable operation across Windows, Linux, network shares, and Docker volume mounts.
- **Four Dedicated Tables**:
  - `sessions`: Session ID, start/end timestamps, duration, average risk, max risk, total alerts, yawns, phone incidents.
  - `trips`: Trip ID, session ID, driver name/email, passenger name/email, vehicle plate, status.
  - `telemetry_logs`: Timestamped records of EAR, MAR, PERCLOS, Yaw, Pitch, Roll, phone presence, and risk score.
  - `alerts`: Detailed logs of every triggered alert, severity level, trigger metric values, and descriptions.

### 13. UI & HUD Overlay (`ui/overlay.py`, `ui/web_app.py`, `ui/templates/index.html`)
- **In-Cabin HUD Canvas**:
  - Draws delicate cyan contours over both eyes and yellow dots over the pupils/irises.
  - Draws magenta contours tracing lip movements.
  - **3D Head Pose Axes**: Projects a 3-axis coordinate frame directly from the driver's nose tip (Red = X / Yaw, Green = Y / Pitch, Blue = Z / Forward gaze vector).
  - Highlighting bounding boxes for detected phones (red) and bottles (orange) with confidence percentages.
  - Top status bar displaying current Risk Band and vehicle speed.
- **FastAPI Web Application**:
  - `/video_feed`: Low-latency MJPEG video stream.
  - `/ws/telemetry`: WebSocket broadcasting live telemetry at 30 FPS.
  - Responsive dark-mode dashboard displaying live metric cards, interactive map (Leaflet with automatic Google Maps JavaScript API enhancement), Trip Registration modal, and in-cabin SOS button.

---

## 💻 Tech Stack & Dependencies

| Layer | Technologies Used | Purpose |
| :--- | :--- | :--- |
| **Core Language** | Python 3.10 – 3.15 | Backend, computer vision, and machine learning |
| **Computer Vision** | OpenCV (`opencv-python`), MediaPipe | Video capture, 478-point facial mesh, 3D `solvePnP` head pose |
| **Object Detection** | Ultralytics YOLOv8 (`yolov8n.pt`) | Real-time phone and drinking bottle detection |
| **Machine Learning** | Scikit-Learn (`scikit-learn`), NumPy, SciPy | 50-tree Random Forest risk classifier, Euclidean math, EMA smoothing |
| **Web Server & API** | FastAPI, Uvicorn, WebSockets, Jinja2, Pydantic | REST API, MJPEG video streaming, real-time telemetry push |
| **Speech Engine** | `pyttsx3`, `pywin32` (Windows SAPI5), `espeak` (Linux) | 100% offline spoken voice warnings per mistake |
| **Maps & Navigation** | Leaflet.js, OpenStreetMap, Google Maps Platform | Live vehicle tracking, breadcrumb trail, Safe Haven routing |
| **Database** | SQLite 3 (`data/driver_logs.db`) | Local persistence of trips, sessions, telemetry, and alerts |
| **Testing** | Pytest (`pytest`) | Automated unit and integration test suite (20 tests) |
| **Containerization** | Docker, Docker Compose | Multi-platform headless deployment |

---

## 🚀 Getting Started (Step-by-Step)

### Step 1: Clone or Open the Repository
Open a terminal in the project directory:
```bash
cd "AI-Driver Monitoring System"
```

### Step 2: Set Up Python Environment
Ensure you have **Python 3.10 to 3.15** installed. We recommend creating a virtual environment:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install all required libraries with a single command:
```bash
pip install -r requirements.txt
```

> **Note for Linux users**: Install system graphics and text-to-speech packages if running on desktop Linux:
> ```bash
> sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 espeak libespeak1
> ```

---

## 🏃 Running the Application

GuardianDrive AI provides multiple convenient ways to run:

### Option A: Web Dashboard (Recommended)
This starts the FastAPI server and automatically opens the interactive dashboard in your default browser:
```bash
python main.py
```
*(Or use the convenience launcher: `python run.py`)*

Once started, open your browser and go to:
👉 **`http://localhost:8000`**

### Option B: Standalone OpenCV Desktop Window
If you prefer a direct OpenCV desktop window without launching a browser:
```bash
python main.py --desktop
```
- Press **`q`** or **`ESC`** in the video window to quit.
- Press **`c`** to calibrate your resting facial baseline (EAR, MAR, Head Pose, Emotion).

### Option C: Run Without Opening Browser Automatically
Useful if running on a remote machine or secondary screen:
```bash
python main.py --no-browser
```

### Option D: Run with Docker & Docker Compose
To run GuardianDrive AI inside a containerized environment:
```bash
docker-compose up --build
```
The web dashboard will be accessible at `http://localhost:8000`.

---

## 🎯 Driver Baseline Calibration

Every human face and vehicle seating position is slightly different. A driver sitting in a tall SUV has a different resting head pitch than a driver in a low sports coupe.

You can calibrate GuardianDrive AI to your specific resting face in 2 seconds:

1. Look straight ahead at the road in your natural, comfortable driving posture.
2. Ensure your face is relaxed (not smiling or frowning).
3. **Calibrate**:
   - In **Web Mode**: Click the **"Calibrate Baseline"** button on the dashboard.
   - In **Desktop Mode**: Press the **`c`** key on your keyboard.
4. The system locks your resting EAR, MAR, neutral head angle offsets, and relaxed eyebrow geometry. All alerts are now tailored specifically to your facial structure!

---

## ⚙️ Configuration Reference (`config.py` & `.env`)

All settings have calibrated defaults that work out-of-the-box. If you wish to customize thresholds, email settings, or Google Maps keys, copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Key Configuration Parameters

| Category | Variable Name | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Camera** | `CAMERA_INDEX` | `0` | Physical camera hardware index (tries `0` and `1`) |
| | `FRAME_WIDTH` / `HEIGHT` | `640` / `480` | Frame resolution downscaled for optimal CPU speed |
| | `TARGET_FPS` | `30` | Target camera acquisition framerate |
| **Drowsiness** | `EAR_THRESHOLD` | `0.22` | Eye Aspect Ratio below this counts as eye closed |
| | `EAR_CONSEC_FRAMES` | `15` | Consecutive frames (~0.5s) to trigger eye closure |
| | `MICROSLEEP_TIME_SEC` | `1.5` | Continuous closure duration triggering emergency alarm |
| | `MAR_THRESHOLD` | `0.45` | Mouth Aspect Ratio above this counts as yawn |
| | `MAR_CONSEC_FRAMES` | `60` | Sustained frames (~2.0s) to confirm genuine yawn |
| | `YAWN_COOLDOWN_SEC` | `3.0` | Debounce time between successive yawns |
| | `PERCLOS_FATIGUE_THRESHOLD` | `0.15` | Rolling 60s eye closure >15% triggers fatigue warning |
| **Distraction** | `YAW_LIMIT_DEG` | `22.0°` | Head turned left/right beyond ±22° |
| | `PITCH_LIMIT_DEG` | `18.0°` | Head tilted up/down beyond ±18° |
| | `ROLL_LIMIT_DEG` | `25.0°` | Head tilted sideways beyond ±25° |
| | `DISTRACTION_TIME_SEC` | `1.5` | Off-road attention duration triggering alert |
| **YOLOv8** | `YOLO_CONF_THRESHOLD` | `0.25` | Lower threshold catches tilted phones against ear |
| | `YOLO_INFER_IMGSZ` | `448` | Higher resolution improves recall on rotated objects |
| | `PHONE_HOLD_TIME_SEC` | `1.0` | Sustained phone detection threshold |
| | `BOTTLE_HOLD_TIME_SEC` | `1.0` | Sustained bottle detection threshold |
| | `PHONE_PROXIMITY_RATIO` | `1.6` | Driver face bounding box proximity multiplier |
| **Risk Weights** | `WEIGHT_EAR` | `0.20` | Weight for instantaneous eye closure |
| | `WEIGHT_PERCLOS` | `0.25` | Weight for rolling 60s cumulative fatigue |
| | `WEIGHT_YAWN` | `0.15` | Weight for recent yawn frequency |
| | `WEIGHT_DISTRACT` | `0.20` | Weight for head pose / gaze off-road |
| | `WEIGHT_PHONE` | `0.20` | Weight for cell phone / bottle interaction |
| **Risk Bands** | `RISK_BAND_LOW_MAX` | `0.35` | [0.00 – 0.35] = Low Risk (Green) |
| | `RISK_BAND_MED_MAX` | `0.70` | (0.35 – 0.70] = Medium Risk (Yellow), >0.70 = High (Red) |
| **Voice Alerts** | `DMS_VOICE_ENABLED` | `true` | Enables offline OS speech synthesis |
| | `VOICE_RATE` | `175` | Words spoken per minute |
| | `VOICE_COOLDOWN_SEC` | `4.0` | Gap between repeated voice warnings of same type |
| **Email (SMTP)** | `DMS_EMAIL_ENABLED` | `false` | Set `true` to enable SMTP emergency emails |
| | `DMS_SMTP_HOST` | `smtp.gmail.com` | SMTP outgoing mail server |
| | `DMS_SMTP_PORT` | `587` | TLS port for outgoing mail |
| | `DMS_SMTP_USER` | `""` | Sender email address |
| | `DMS_SMTP_PASSWORD` | `""` | 16-character Gmail App Password |
| | `DMS_EMAIL_ONLY_ON_SOS`| `true` | Restricts email dispatch strictly to SOS triggers |
| **GPS & Maps** | `DMS_GPS_MODE` | `simulated` | `"simulated"` or `"ip"` |
| | `GOOGLE_MAPS_API_KEY` | `""` | Optional Google Maps API key (Leaflet used if blank)|

---

## 📡 REST API & WebSocket Reference

The FastAPI backend exposes comprehensive REST endpoints and real-time WebSockets:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main HTML5 dashboard interface |
| `GET` | `/video_feed` | Streams live MJPEG video with HUD overlay annotations |
| `WS` | `/ws/telemetry` | Real-time WebSocket broadcasting full driver telemetry |
| `POST` | `/api/trip/start` | Registers driver, passenger, and emergency email for the trip |
| `GET` | `/api/trip/current` | Returns currently registered trip and occupant particulars |
| `POST` | `/api/emergency/sos` | **In-Cabin SOS Trigger**: Announces spoken SOS & sends emergency email |
| `POST` | `/api/test_alert` | Manually triggers a test alert cycle (voice + email) |
| `POST` | `/api/calibrate_driver`| Calibrates resting EAR, MAR, head pose zero, and emotion |
| `GET` | `/api/session/history` | Returns recent 60 telemetry samples and last 20 incident alerts |
| `GET` | `/api/maps/config` | Returns Google Maps configuration, status, and vehicle coordinates |
| `GET` | `/api/maps/nearby_havens`| Finds closest Rest Areas, Fuel Stations, and Hospitals |
| `POST` | `/api/gps/update` | Ingests authentic live coordinates streamed from client device |
| `GET` | `/api/maps/route` | Returns recorded GPS route breadcrumb history (up to 200 points) |
| `GET` | `/api/reports/{timeframe}`| Returns driving scorecard JSON (`daily`, `weekly`, `monthly`) |
| `GET` | `/api/reports_view/html` | Renders a printable executive HTML driving scorecard |

---

## 🗄️ Database Schema & Inspection

All session statistics, trip registration details, sampled telemetry, and incident alerts are stored in a local SQLite file:  
📁 **[`data/driver_logs.db`](data/driver_logs.db)**

### Table 1: `trips`
| Column | Type | Description |
| :--- | :--- | :--- |
| `trip_id` | TEXT PRIMARY KEY | Unique trip identifier (e.g. `TRIP-1718000000`) |
| `session_id` | TEXT | Linked session ID |
| `driver_name` | TEXT | Registered driver name |
| `driver_email` | TEXT | Driver email address |
| `passenger_name` | TEXT | Registered passenger name |
| `passenger_emergency_email` | TEXT | Primary recipient for emergency alerts |
| `vehicle_id` | TEXT | Vehicle registration / license plate |
| `start_time` / `end_time` | REAL | Epoch timestamps |
| `status` | TEXT | `ACTIVE` or `COMPLETED` |

### Table 2: `sessions`
| Column | Type | Description |
| :--- | :--- | :--- |
| `session_id` | TEXT PRIMARY KEY | Unique execution session identifier |
| `start_time` / `end_time` | REAL | Start and completion timestamps |
| `duration_sec` | REAL | Total driving duration in seconds |
| `avg_risk` / `max_risk` | REAL | Average and peak calculated risk score |
| `total_alerts` | INTEGER | Total safety alerts recorded during the drive |
| `total_yawns` | INTEGER | Number of confirmed yawning events |
| `phone_events` | INTEGER | Number of mobile phone usage incidents |

### Table 3: `telemetry_logs`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY | Autoincrementing record ID |
| `session_id` | TEXT | Foreign key referencing `sessions.session_id` |
| `timestamp` | REAL | Epoch timestamp of telemetry sample |
| `ear` / `mar` / `perclos` | REAL | Physiological eye and mouth measurements |
| `yaw` / `pitch` / `roll` | REAL | 3D head pose angles in degrees |
| `phone_detected` | INTEGER | Boolean flag (1 if phone held, 0 if clear) |
| `risk_score` | REAL | Combined hybrid safety risk score (0.00 – 1.00) |
| `risk_level` | TEXT | Categorical band (`Low`, `Medium`, `High`) |

### Table 4: `alerts`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY | Autoincrementing record ID |
| `session_id` | TEXT | Foreign key referencing `sessions.session_id` |
| `timestamp` | REAL | Epoch timestamp when alert was triggered |
| `alert_type` | TEXT | Specific alert key (e.g. `DISTRACTION_LOOKING_LEFT`) |
| `severity` | TEXT | `MEDIUM`, `HIGH`, or `CRITICAL` |
| `trigger_value` | REAL | Metric value that triggered the threshold |
| `details` | TEXT | Human-readable explanation and GPS context |

### How to Inspect the Database

1. **Inside VS Code / Cursor (Easiest)**: Install the free **SQLite Viewer** extension (`qwtel.sqlite-viewer`), then click on `data/driver_logs.db` in your file tree to browse tables like an Excel spreadsheet.
2. **Desktop GUI**: Open `data/driver_logs.db` in [DB Browser for SQLite](https://sqlitebrowser.org/dl/).
3. **From Terminal**:
   ```bash
   python -c "import sqlite3; conn = sqlite3.connect('data/driver_logs.db'); print(conn.execute('SELECT alert_type, severity, details FROM alerts ORDER BY id DESC LIMIT 5').fetchall())"
   ```
4. **Via Browser**: Navigate to `http://localhost:8000/api/reports_view/html` for an executive summary generated directly from the database.

---

## 🧪 Automated Testing Suite

GuardianDrive AI comes with a rigorous automated test suite containing **20 unit and integration tests** using `pytest`.

To execute the entire test suite:
```bash
python -m pytest tests/ -v
```

### What the Tests Verify:

- **`tests/test_advanced_features.py`**:
  - `test_haversine_and_google_maps_service`: Validates great-circle distance formulas and breadcrumb history tracking.
  - `test_emotion_and_stress_detector`: Tests FACS geometric brow furrowing, mouth tension, and classification into Calm, Stressed, Fatigued, and Frustrated.
  - `test_report_generator_timeframes`: Validates scorecard generation across 24-hour, 7-day, and 30-day windows.
  - `test_high_accuracy_calibration_and_smoothing`: Tests baseline calibration locking and exponential moving average smoothing.
  - `test_live_speed_and_stationary_zero_telemetry`: Verifies that stationary vehicles report strictly $0.0\text{ km/h}$ and filter out GPS jitter.
  - `test_immediate_email_formatting_and_sos`: Verifies immediate email composition containing live coordinates and Google Maps links.
  - `test_trip_registration_and_db_persistence`: Confirms that trip occupant particulars and sessions persist accurately in SQLite.
  - `test_traditional_email_layout_and_passenger_emergency_routing`: Confirms dual email routing to passenger emergency contacts.
  - `test_safe_havens_non_blocking_performance`: Ensures nearest safe haven queries execute quickly without blocking the video thread.
  - `test_gaze_suppression_on_closed_eyes`: Verifies that natural eye blinks suppress false downward gaze alarms.
  - `test_html_escaping_in_advisory_and_report`: Tests strict sanitization against XSS in email bodies and reports.
  - `test_email_only_on_sos_button`: Verifies that regular alerts do not trigger emails when `EMAIL_ONLY_ON_SOS=true`.

- **`tests/test_metrics.py`**:
  - `test_ear_calculation_open_and_closed`: Verifies Eye Aspect Ratio formulas on geometric open and closed eye coordinates.
  - `test_mar_calculation_yawning`: Validates Mouth Aspect Ratio calculations during yawning.
  - `test_risk_engine_bounds_and_bands`: Ensures calculated risk scores are strictly bounded in $[0.0, 1.0]$ and map to correct risk bands.
  - `test_head_pose_variance`: Tests 10-second sliding window variance calculations for head movement.
  - `test_ml_risk_classifier`: Tests the Scikit-Learn Random Forest probabilistic classification on synthetic feature vectors.
  - `test_risk_engine_hybrid_ml_integration`: Verifies fusion between the 5-factor math model and Random Forest probabilities.

- **`tests/test_specific_alerts.py`**:
  - `test_alert_phrases_and_advisories_exist`: Confirms that every alert condition has a corresponding voice phrase and tailored advisory steps.
  - `test_distraction_type_classification`: Verifies directional classification for pitch, yaw, and roll angles.

---

## 🛠️ Real-World Engineering Insights

Building a computer vision system that works inside a real moving car is very different from running a toy model in a notebook. Here are four real challenges we encountered and how we engineered solutions for them:

### 1. The "Blinking vs Looking Down" Problem
In computer vision, calculating gaze pitch depends on the vertical distance between the eyelid and iris. When a human blinks naturally (~150 to 300 ms), the eyelid moves downward, which naive algorithms misinterpret as "driver looking down at their phone". This caused irritating false alarms every few seconds.  
**Our Solution**: In `core/distraction.py`, we implemented a smart gaze-suppression circuit. The moment EAR drops below the closure threshold, gaze deviation alerts are temporarily paused until the eyes reopen.

### 2. The "Passenger Checking Instagram" Problem
If you run an off-the-shelf YOLO object detector inside a car cabin, it detects *every* phone in the frame. If your passenger in the front seat checks their phone, conventional systems sound an alarm and penalize the driver.  
**Our Solution**: In `core/phone_detector.py`, we compute the driver's full 468-point face bounding box and construct an active interaction zone. Any cell phone detected outside this perimeter is classified as a passenger device and completely ignored.

### 3. The "Freezing pyttsx3 Thread" Bug
On Windows, `pyttsx3` wraps Microsoft's SAPI5 speech engine. When called repeatedly from a background worker thread using the standard `engine.say()` and `engine.runAndWait()`, the COM thread deadlocks after the first utterance and goes permanently silent.  
**Our Solution**: In `core/voice.py`, we initialize `pythoncom.CoInitialize()` inside the speech worker thread, instantiate a lightweight fresh engine instance for each message, speak the text, cleanly terminate the engine, and garbage-collect it. Result: 100% reliable repeated voice alerts.

### 4. The "Parked Car GPS Drift" Problem
Standard GPS sensors constantly jitter by 1 to 5 meters even when a car is parked in a driveway. Naive velocity calculations ($v = \Delta d / \Delta t$) divide this jitter by small time deltas, causing stationary dashboards to report erratic phantom speeds like $8\text{ km/h}$ or $14\text{ km/h}$.  
**Our Solution**: In `core/maps_service.py`, we built a spatial jitter deadband. Any coordinate shift smaller than 3.0 meters while stationary is filtered out as sensor noise, locking reported speed strictly to $0.0\text{ km/h}$.

---

## ❓ Frequently Asked Questions (FAQ) & Troubleshooting

### Q: Why does my webcam show a black screen or fail to connect?
**A**: Ensure no other application (Zoom, Teams, Skype, or browser) is currently using your webcam. If you have multiple cameras (e.g. an integrated laptop camera and an external USB webcam), change `CAMERA_INDEX = 1` in `config.py` or `.env`.

### Q: Do I need an internet connection to run this?
**A**: **No.** All face tracking, head pose estimation, YOLO phone/bottle detection, Random Forest risk scoring, voice alerts, and SQLite database logging run 100% offline. Internet is only used if you choose to enable SMTP email alerts or load live map tiles. If offline, the map gracefully displays an offline coordinate grid.

### Q: How do I set up Gmail for emergency emails?
**A**: Google requires an **App Password** rather than your regular account password:
1. Go to your **Google Account** $\rightarrow$ **Security**.
2. Under *"How you sign in to Google"*, enable **2-Step Verification**.
3. Search for **App passwords**.
4. Generate a 16-character App Password (e.g., `abcd efgh ijkl mnop`).
5. Paste it into your `.env` file as `DMS_SMTP_PASSWORD=abcdefghijklmnop` and set `DMS_EMAIL_ENABLED=true`.

### Q: Can I run this on a Raspberry Pi or Jetson Nano?
**A**: Yes! The MediaPipe Face Mesh and YOLOv8 Nano architectures are lightweight. On embedded ARM platforms (Raspberry Pi 4/5, NVIDIA Jetson), GuardianDrive AI runs efficiently.

---

## 📄 License & Acknowledgments

This project is open-source under the **MIT License**.

- **MediaPipe**: Google's open-source cross-platform machine learning framework for live perceptual computing.
- **Ultralytics YOLOv8**: High-speed real-time object detection models.
- **OpenCV**: Open Source Computer Vision Library.
- **FastAPI**: Modern, fast web framework for building APIs with Python.
- **Leaflet & OpenStreetMap**: Universal open-source interactive mapping.

---

<div align="center">
  <b>Built with ❤️ for driver safety and accident prevention. Drive safe!</b>
</div>
