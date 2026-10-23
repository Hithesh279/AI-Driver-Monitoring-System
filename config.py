"""
AI-Powered Driver Monitoring & Safety Assistant
Central Configuration & Calibrated Thresholds
"""

import os
import warnings

# Suppress Protobuf, TensorFlow, MediaPipe, and Intel OpenMP shutdown noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

# Load .env file if present (no error if missing / python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "driver_logs.db"

# ----------------------------------------------------------------------
# 1. Camera & Video Pipeline Settings
# ----------------------------------------------------------------------
CAMERA_INDEX = 0             # Default webcam index
FRAME_WIDTH = 640            # Downscaled width for optimal CPU performance
FRAME_HEIGHT = 480           # Downscaled height
TARGET_FPS = 30              # Target framerate
JPEG_QUALITY = 80            # Compression quality for web streaming

# ----------------------------------------------------------------------
# 2. MediaPipe Face Mesh Landmark Settings
# ----------------------------------------------------------------------
REFINE_LANDMARKS = True      # 478 landmarks (enables iris tracking 468-477)
MAX_NUM_FACES = 1            # Single driver tracking
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Landmark Index Sets (MediaPipe 468 + 10 Iris landmarks)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]
RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]
LIPS_LANDMARKS = [61, 291, 0, 17, 13, 14, 78, 308] # Outer and inner lip anchors

# ----------------------------------------------------------------------
# 3. Drowsiness & Fatigue Thresholds
# ----------------------------------------------------------------------
EAR_THRESHOLD = 0.22         # Eye Aspect Ratio below this = eye closed
EAR_CONSEC_FRAMES = 15       # Consecutive frames below threshold (~0.5s) to trigger eye closure
MICROSLEEP_TIME_SEC = 1.5    # Sustained closure triggering critical microsleep alarm

MAR_THRESHOLD = 0.45         # Mouth Aspect Ratio above this = yawn open mouth (calibrated)
MAR_CONSEC_FRAMES = 60       # Sustained frames to confirm yawn event (~2.0s at 30 FPS)
YAWN_COOLDOWN_SEC = 3.0      # Debounce period between distinct yawn counts

PERCLOS_WINDOW_SECONDS = 60  # Rolling window for Percentage of Eye Closure
PERCLOS_FATIGUE_THRESHOLD = 0.15  # PERCLOS > 15% indicates significant physiological fatigue

# ----------------------------------------------------------------------
# 4. Distraction & Head Pose Thresholds (solvePnP)
# ----------------------------------------------------------------------
YAW_LIMIT_DEG = 22.0         # Head turned left/right beyond +/- 22 degrees
PITCH_LIMIT_DEG = 18.0       # Head tilted up/down beyond +/- 18 degrees
ROLL_LIMIT_DEG = 25.0        # Head tilt
GAZE_DEVIATION_THRESHOLD = 0.12 # Gaze offset from forward center (calibrated iris offset)
DISTRACTION_TIME_SEC = 1.5   # Continuous off-road attention duration triggering alert

# ----------------------------------------------------------------------
# 5. YOLOv8 Object Detection (Cell Phone)
# ----------------------------------------------------------------------
YOLO_MODEL_NAME = "yolov8n.pt"  # Lightweight nano model
YOLO_PHONE_CLASS_ID = 67        # COCO class index for 'cell phone'
YOLO_BOTTLE_CLASS_ID = 39       # COCO class index for 'bottle' (drinking-while-driving)
YOLO_CONF_THRESHOLD = 0.25       # Lowered from 0.40 — catches tilted/partly-occluded phones (near ear) too
YOLO_INFER_IMGSZ = 448           # Raised from 320 — better recall on small/rotated objects
YOLO_FRAME_INTERVAL = 3          # Raised sampling rate (was 5) so a brief tilt doesn't get missed
PHONE_HOLD_TIME_SEC = 1.0        # Sustained phone detection threshold
BOTTLE_HOLD_TIME_SEC = 1.0       # Sustained bottle-near-face detection threshold

# ----------------------------------------------------------------------
# 6. Transparent 5-Factor Risk Weighting Model
# ----------------------------------------------------------------------
# Risk Score = w_ear*f_ear + w_perclos*f_perclos + w_yawn*f_yawn + w_distract*f_distract + w_phone*f_phone
WEIGHT_EAR = 0.20            # Instantaneous eye closure penalty
WEIGHT_PERCLOS = 0.25        # 60-second cumulative fatigue penalty
WEIGHT_YAWN = 0.15           # Recent yawn frequency penalty
WEIGHT_DISTRACT = 0.20       # Head pose / gaze off-road penalty
WEIGHT_PHONE = 0.20          # Cell phone interaction penalty

# Config-Driven Risk Band Boundaries
RISK_BAND_LOW_MAX = 0.35     # [0.00 - 0.35] -> Low (Safe / Normal)
RISK_BAND_MED_MAX = 0.70     # (0.35 - 0.70] -> Medium (Caution / Advisory)
# (0.70 - 1.00] -> High (Critical / Intervention)

# ----------------------------------------------------------------------
# 7. Web & Server Settings
# ----------------------------------------------------------------------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# ----------------------------------------------------------------------
# 8. Email Alert Settings (SMTP) — set via environment variables, never hardcode
# ----------------------------------------------------------------------
EMAIL_ENABLED = os.getenv("DMS_EMAIL_ENABLED", "false").lower() == "true"
SMTP_HOST = os.getenv("DMS_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("DMS_SMTP_PORT", "587"))
SMTP_USER = os.getenv("DMS_SMTP_USER", "").strip().strip('"').strip("'")
_raw_pw = os.getenv("DMS_SMTP_PASSWORD", "").strip().strip('"').strip("'")
if "gmail" in SMTP_HOST.lower():
    # Gmail App Passwords are 16 letters, often copied with spaces e.g. 'abcd efgh ijkl mnop'
    SMTP_PASSWORD = _raw_pw.replace(" ", "").replace("-", "")
else:
    SMTP_PASSWORD = _raw_pw
ALERT_RECIPIENT_EMAILS = [
    e.strip() for e in os.getenv("DMS_ALERT_RECIPIENTS", "").split(",") if e.strip()
]
EMAIL_COOLDOWN_SEC = int(os.getenv("EMAIL_COOLDOWN_SEC", os.getenv("DMS_EMAIL_COOLDOWN_SEC", "60")))      # Minimum gap between two emails, to avoid spamming
EMAIL_ONLY_ON_SOS = os.getenv("DMS_EMAIL_ONLY_ON_SOS", "true").lower() == "true"

# ----------------------------------------------------------------------
# 9. GPS Tracking Settings
# ----------------------------------------------------------------------
# GPS_MODE: "simulated" (default, no hardware needed) or "ip" (approx. location via IP)
GPS_MODE = os.getenv("DMS_GPS_MODE", "simulated")
GPS_SIM_START_LAT = float(os.getenv("DMS_GPS_LAT", "14.4673"))   # Kadapa, AP as a default demo point
GPS_SIM_START_LON = float(os.getenv("DMS_GPS_LON", "78.8242"))

# ----------------------------------------------------------------------
# 10. Voice Alert Settings (offline TTS, no training/API key needed)
# ----------------------------------------------------------------------
VOICE_ALERTS_ENABLED = os.getenv("DMS_VOICE_ENABLED", "true").lower() == "true"
VOICE_RATE = 175             # Words per minute
VOICE_COOLDOWN_SEC = 4       # Repeats every 4s per alert type while condition stays active

# ----------------------------------------------------------------------
# 11. Google Maps Platform Integration Settings
# ----------------------------------------------------------------------
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_MAPS_MAP_ID = os.getenv("GOOGLE_MAPS_MAP_ID", "DEMO_MAP_ID")
GOOGLE_MAPS_ZOOM = int(os.getenv("GOOGLE_MAPS_ZOOM", "15"))
MAPS_BREADCRUMB_LIMIT = 200   # Max GPS route trail waypoints stored in memory
SAFE_HAVEN_SEARCH_RADIUS_KM = 15.0  # Search radius for emergency rest stops / hospitals

# ----------------------------------------------------------------------
# 12. High-Accuracy Calibration & Temporal Smoothing
# ----------------------------------------------------------------------
SMOOTHING_ALPHA = 0.65       # Exponential moving average filter coefficient (0.0=frozen, 1.0=raw)
CALIBRATION_WINDOW_SEC = 2.0 # Calibration baseline acquisition duration
PHONE_PROXIMITY_RATIO = 1.6  # Max distance from face center / bbox for driver phone classification

# ----------------------------------------------------------------------
# 13. Emotion & Stress Recognition Settings (PPT Module 03)
# ----------------------------------------------------------------------
STRESS_THRESHOLD = 0.50      # Stress index above this contributes to elevated risk
EMOTION_FRAME_INTERVAL = 2   # Compute emotion every N frames for low CPU overhead

