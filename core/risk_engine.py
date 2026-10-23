"""
ML-Powered & 5-Factor Fused Driver Risk Assessment Engine:
Combines Scikit-Learn Random Forest Classifier (PPT Slide 10 & 14)
with transparent 5-Factor mathematical explainability.
Includes a pure-NumPy ensemble fallback if scikit-learn is not installed.
Features: Blink rate, PERCLOS, Yawn frequency, Head-pose variance, Phone duration, Stress index.
"""

import pickle
import warnings
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    sklearn = None
    RandomForestClassifier = None
    HAS_SKLEARN = False

from config import (
    WEIGHT_EAR,
    WEIGHT_PERCLOS,
    WEIGHT_YAWN,
    WEIGHT_DISTRACT,
    WEIGHT_PHONE,
    RISK_BAND_LOW_MAX,
    RISK_BAND_MED_MAX,
    MAR_THRESHOLD,
    MICROSLEEP_TIME_SEC,
    PERCLOS_FATIGUE_THRESHOLD,
    DISTRACTION_TIME_SEC,
    PHONE_HOLD_TIME_SEC,
    BOTTLE_HOLD_TIME_SEC
)

MODEL_FILE = Path(__file__).resolve().parent.parent / "data" / "risk_rf_model.pkl"


class _NumpyFallbackForest:
    """
    Lightweight pure-NumPy probabilistic risk classifier used as a fallback
    if scikit-learn is not installed in the active Python environment.
    """
    def __init__(self):
        self.feature_importances_ = np.array([0.12, 0.28, 0.14, 0.16, 0.18, 0.12], dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = []
        for row in X:
            blink, perclos, yawn, pose_var, phone_dur, stress = [float(v) for v in row]
            blink_strain = 1.0 if (blink < 8.0 or blink > 28.0) else (0.4 if (blink < 11.0 or blink > 23.0) else 0.0)
            p_norm = min(1.0, perclos / 0.20)
            y_norm = min(1.0, yawn / 3.0)
            v_norm = min(1.0, pose_var / 70.0)
            ph_norm = min(1.0, phone_dur / 2.0)
            st_norm = min(1.0, stress)

            hazard = (
                0.12 * blink_strain +
                0.28 * p_norm +
                0.14 * y_norm +
                0.16 * v_norm +
                0.18 * ph_norm +
                0.12 * st_norm
            )
            if perclos >= 0.22 or phone_dur >= 2.0:
                hazard = max(hazard, 0.78)
            elif perclos >= 0.10 or phone_dur >= 0.4 or pose_var >= 30.0 or stress >= 0.55:
                hazard = max(hazard, 0.45)

            s_low = max(0.05, 1.0 - hazard * 1.6)
            s_med = max(0.05, 1.0 - abs(hazard - 0.50) * 2.2)
            s_high = max(0.02, (hazard - 0.25) * 1.5) if hazard > 0.25 else 0.02
            logits = np.array([s_low, s_med, s_high], dtype=np.float64)
            exp_l = np.exp(logits * 3.0)
            probs = exp_l / np.sum(exp_l)
            out.append(probs)
        return np.array(out, dtype=np.float64)


class MLRiskClassifier:
    """
    Random Forest Risk Classifier (scikit-learn) trained on structured behavioral features (PPT Slide 10).
    Maps behavioral signals -> Low, Medium, High risk classes with probabilistic confidence.
    """
    FEATURE_NAMES = [
        "blink_rate",
        "perclos",
        "yawn_frequency",
        "head_pose_variance",
        "phone_duration",
        "stress_score"
    ]
    CLASSES = ["Low", "Medium", "High"]

    def __init__(self, model_path: Path = MODEL_FILE):
        self.model_path = model_path
        self.model: Optional[Any] = None
        self._load_or_train()

    def _generate_synthetic_training_data(self, n_samples_per_class: int = 500):
        rng = np.random.RandomState(42)
        X = []
        y = []

        # Class 0: Low Risk (Safe, attentive driving)
        for _ in range(n_samples_per_class):
            blink_rate = rng.uniform(12.0, 22.0)
            perclos = rng.uniform(0.01, 0.08)
            yawn_freq = float(rng.choice([0, 0, 0, 1]))
            head_pose_var = rng.uniform(1.0, 15.0)
            phone_dur = 0.0
            stress = rng.uniform(0.02, 0.25)
            X.append([blink_rate, perclos, yawn_freq, head_pose_var, phone_dur, stress])
            y.append(0)

        # Class 1: Medium Risk (Caution: mild fatigue, gaze drift, or restlessness)
        for _ in range(n_samples_per_class):
            mode = rng.choice(["fatigue", "distraction", "stress"])
            if mode == "fatigue":
                blink_rate = rng.uniform(6.0, 12.0)
                perclos = rng.uniform(0.09, 0.16)
                yawn_freq = float(rng.choice([1, 2, 3]))
                head_pose_var = rng.uniform(10.0, 35.0)
                phone_dur = 0.0
                stress = rng.uniform(0.20, 0.50)
            elif mode == "distraction":
                blink_rate = rng.uniform(14.0, 24.0)
                perclos = rng.uniform(0.02, 0.09)
                yawn_freq = float(rng.choice([0, 1]))
                head_pose_var = rng.uniform(25.0, 60.0)
                phone_dur = rng.uniform(0.2, 1.2)
                stress = rng.uniform(0.15, 0.45)
            else:
                blink_rate = rng.uniform(22.0, 35.0)
                perclos = rng.uniform(0.03, 0.10)
                yawn_freq = 0.0
                head_pose_var = rng.uniform(15.0, 40.0)
                phone_dur = 0.0
                stress = rng.uniform(0.55, 0.75)
            X.append([blink_rate, perclos, yawn_freq, head_pose_var, phone_dur, stress])
            y.append(1)

        # Class 2: High Risk (Critical Danger: microsleep, active phone, sustained distraction)
        for _ in range(n_samples_per_class):
            mode = rng.choice(["microsleep", "phone", "severe_distraction", "severe_fatigue"])
            if mode == "microsleep":
                blink_rate = rng.uniform(2.0, 8.0)
                perclos = rng.uniform(0.20, 0.85)
                yawn_freq = float(rng.choice([2, 3, 4, 5]))
                head_pose_var = rng.uniform(40.0, 120.0)
                phone_dur = 0.0
                stress = rng.uniform(0.40, 0.80)
            elif mode == "phone":
                blink_rate = rng.uniform(10.0, 20.0)
                perclos = rng.uniform(0.04, 0.14)
                yawn_freq = float(rng.choice([0, 1]))
                head_pose_var = rng.uniform(30.0, 90.0)
                phone_dur = rng.uniform(2.0, 10.0)
                stress = rng.uniform(0.30, 0.70)
            elif mode == "severe_distraction":
                blink_rate = rng.uniform(12.0, 25.0)
                perclos = rng.uniform(0.05, 0.15)
                yawn_freq = 0.0
                head_pose_var = rng.uniform(65.0, 180.0)
                phone_dur = rng.uniform(0.5, 4.0)
                stress = rng.uniform(0.60, 0.95)
            else:
                blink_rate = rng.uniform(4.0, 9.0)
                perclos = rng.uniform(0.18, 0.45)
                yawn_freq = float(rng.choice([3, 4, 6]))
                head_pose_var = rng.uniform(35.0, 100.0)
                phone_dur = 0.0
                stress = rng.uniform(0.65, 0.90)
            X.append([blink_rate, perclos, yawn_freq, head_pose_var, phone_dur, stress])
            y.append(2)

        return np.array(X), np.array(y)

    def _load_or_train(self):
        if not HAS_SKLEARN:
            self.model = _NumpyFallbackForest()
            return

        if self.model_path.exists():
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with open(self.model_path, "rb") as f:
                        loaded = pickle.load(f)
                    version_mismatch = any("InconsistentVersionWarning" in str(w.category) for w in caught)
                    if not version_mismatch and hasattr(loaded, "predict_proba"):
                        self.model = loaded
                        return
            except Exception as e:
                print(f"[MLRiskClassifier] Could not load {self.model_path} ({e}). Retraining...")

        # Train Random Forest Classifier (PPT Slide 10 & 14, optimized 50-tree ensemble for <= 2ms latency)
        X, y = self._generate_synthetic_training_data(n_samples_per_class=800)
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        rf.fit(X, y)
        self.model = rf
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(rf, f)
            print(f"[MLRiskClassifier] Trained and cached Random Forest model (sklearn {sklearn.__version__}) to {self.model_path}")
        except Exception as e:
            print(f"[MLRiskClassifier] Cache save warning ({e})")

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        features = [
            float(feature_dict.get("blink_rate", 18.0)),
            float(feature_dict.get("perclos", 0.0)),
            float(feature_dict.get("yawn_frequency", 0.0)),
            float(feature_dict.get("head_pose_variance", 0.0)),
            float(feature_dict.get("phone_duration", 0.0)),
            float(feature_dict.get("stress_score", 0.0))
        ]
        X = np.array([features])
        probs = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.CLASSES[pred_idx]

        importances = dict(zip(self.FEATURE_NAMES, [round(float(w), 3) for w in self.model.feature_importances_]))

        return {
            "predicted_class": pred_class,
            "level": pred_class,
            "confidence": round(float(probs[pred_idx]), 3),
            "probability": round(float(probs[pred_idx]), 3),
            "probabilities": {
                "Low": round(float(probs[0]), 3),
                "Medium": round(float(probs[1]), 3),
                "High": round(float(probs[2]), 3)
            },
            "feature_importances": importances,
            "features_extracted": dict(zip(self.FEATURE_NAMES, [round(f, 3) for f in features]))
        }


class RiskEngine:
    """
    Hybrid Risk Engine combining Scikit-Learn Random Forest Classifier (PPT Slide 10)
    with the transparent 5-Factor mathematical model.
    """
    def __init__(self):
        self.w_ear = WEIGHT_EAR
        self.w_perclos = WEIGHT_PERCLOS
        self.w_yawn = WEIGHT_YAWN
        self.w_distract = WEIGHT_DISTRACT
        self.w_phone = WEIGHT_PHONE
        self.ml_classifier = MLRiskClassifier()

    def calculate_factors(self,
                          drowsiness_data: Dict[str, Any],
                          distraction_data: Dict[str, Any],
                          phone_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes normalized factor values f in [0.0, 1.0] for each behavioral signal.
        """
        # 1. EAR Factor (instantaneous eye closure duration)
        if drowsiness_data.get("is_eye_closed", False):
            dur = drowsiness_data.get("closure_duration_sec", 0.0)
            f_ear = min(1.0, dur / max(0.1, MICROSLEEP_TIME_SEC))
        else:
            f_ear = 0.0

        # 2. PERCLOS Factor (cumulative fatigue over rolling 60s)
        perclos = drowsiness_data.get("perclos", 0.0)
        f_perclos = min(1.0, perclos / max(0.01, PERCLOS_FATIGUE_THRESHOLD))

        # 3. Yawn Factor (active yawn + recent yawn frequency)
        is_yawn = drowsiness_data.get("is_yawning", False)
        recent_yawns = drowsiness_data.get("recent_yawns", 0)
        
        f_yawn_active = 0.75 if is_yawn else 0.0
        f_yawn_freq = min(1.0, recent_yawns * 0.25)
        f_yawn = min(1.0, f_yawn_active + f_yawn_freq)

        # 4. Distraction Factor (head pose yaw/pitch and gaze off-road duration)
        dist_dur = distraction_data.get("distraction_duration_sec", 0.0)
        if distraction_data.get("head_off_road", False) or distraction_data.get("gaze_off_road", False):
            f_distract = min(1.0, dist_dur / max(0.1, DISTRACTION_TIME_SEC))
        else:
            f_distract = 0.0

        # 5. Handheld Object Factor (phone presence/hold duration + bottle drinking)
        f_phone = 0.0
        if phone_data.get("phone_detected", False):
            conf = phone_data.get("confidence", 0.5)
            phone_dur = phone_data.get("phone_duration_sec", 0.0)
            dur_mult = min(1.0, phone_dur / max(0.1, PHONE_HOLD_TIME_SEC))
            f_phone = min(1.0, conf * (0.5 + 0.5 * dur_mult))
        if phone_data.get("bottle_detected", False):
            b_conf = phone_data.get("bottle_confidence", 0.5)
            b_dur = phone_data.get("bottle_duration_sec", 0.0)
            b_mult = min(1.0, b_dur / max(0.1, BOTTLE_HOLD_TIME_SEC))
            f_bottle = min(0.85, b_conf * (0.4 + 0.45 * b_mult))
            f_phone = max(f_phone, f_bottle)

        return {
            "f_ear": float(f_ear),
            "f_perclos": float(f_perclos),
            "f_yawn": float(f_yawn),
            "f_distract": float(f_distract),
            "f_phone": float(f_phone)
        }

    def evaluate(self,
                 drowsiness_data: Dict[str, Any],
                 distraction_data: Dict[str, Any],
                 phone_data: Dict[str, Any],
                 emotion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fuses behavioral factors into total risk score and classifies using both Random Forest ML and math rules.
        """
        factors = self.calculate_factors(drowsiness_data, distraction_data, phone_data)

        # 1. Mathematical Weighted Score
        raw_score = (
            self.w_ear * factors["f_ear"] +
            self.w_perclos * factors["f_perclos"] +
            self.w_yawn * factors["f_yawn"] +
            self.w_distract * factors["f_distract"] +
            self.w_phone * factors["f_phone"]
        )

        stress_val = 0.0
        if emotion_data:
            stress_val = float(emotion_data.get("stress_score", 0.0))
            if emotion_data.get("is_stressed", False):
                raw_score += (0.10 * stress_val)

        # Life-safety overrides for active hazards
        if drowsiness_data.get("is_microsleep", False):
            raw_score = max(raw_score, 0.85)
        elif drowsiness_data.get("is_drowsy", False) or (drowsiness_data.get("is_eye_closed", False) and drowsiness_data.get("closure_duration_sec", 0.0) > 0.4):
            raw_score = max(raw_score, 0.75)

        if phone_data.get("phone_detected", False) or phone_data.get("is_phone_alert", False):
            raw_score = max(raw_score, 0.75)
        elif phone_data.get("is_bottle_alert", False) or phone_data.get("bottle_detected", False):
            raw_score = max(raw_score, 0.52)

        if distraction_data.get("is_distracted", False) and phone_data.get("is_phone_alert", False):
            raw_score = max(raw_score, 0.90)
        elif distraction_data.get("is_distracted", False):
            raw_score = max(raw_score, 0.48)
        elif distraction_data.get("head_off_road", False) or distraction_data.get("gaze_off_road", False):
            raw_score = max(raw_score, 0.32)

        if drowsiness_data.get("is_yawning", False) or (drowsiness_data.get("mar", 0.0) >= MAR_THRESHOLD):
            raw_score = max(raw_score, 0.42)

        math_score = float(np.clip(raw_score, 0.0, 1.0))

        # 2. Random Forest ML Classifier (PPT Slide 10 & 14)
        ml_features = {
            "blink_rate": float(drowsiness_data.get("blink_rate_bpm", 18.0)),
            "perclos": float(drowsiness_data.get("perclos", 0.0)),
            "yawn_frequency": float(drowsiness_data.get("recent_yawns", 0.0) + (1.0 if drowsiness_data.get("is_yawning") else 0.0)),
            "head_pose_variance": float(distraction_data.get("head_pose_variance", 0.0)),
            "phone_duration": float(max(phone_data.get("phone_duration_sec", 0.0), phone_data.get("bottle_duration_sec", 0.0) * 0.6)),
            "stress_score": stress_val
        }
        ml_result = self.ml_classifier.predict(ml_features)

        # Hybrid Fused Score (calibrating mathematical score with ML probabilistic confidence)
        ml_high_prob = ml_result["probabilities"]["High"]
        ml_med_prob = ml_result["probabilities"]["Medium"]
        ml_boost = (ml_high_prob * 0.20) + (ml_med_prob * 0.08)
        fused_score = float(np.clip(math_score * 0.80 + ml_boost, 0.0, 1.0))

        # Determine Unified Risk Band
        if fused_score <= RISK_BAND_LOW_MAX:
            level = "Low"
            status_text = "Safe & Attentive"
            color_hex = "#22c55e" # Emerald Green
        elif fused_score <= RISK_BAND_MED_MAX:
            level = "Medium"
            status_text = "Caution / Warning"
            color_hex = "#eab308" # Amber Yellow
        else:
            level = "High"
            status_text = "Critical Danger"
            color_hex = "#ef4444" # Bright Red

        # Identify primary risk contributors
        active_reasons = []
        if factors["f_ear"] > 0.4:
            active_reasons.append("Eye Closure / Micro-sleep")
        if factors["f_perclos"] > 0.4:
            active_reasons.append("Cumulative Fatigue (High PERCLOS)")
        if factors["f_yawn"] > 0.4:
            active_reasons.append("Frequent Yawning")
        if factors["f_distract"] > 0.4:
            active_reasons.append("Off-road Head/Gaze Deviation")
        if phone_data.get("phone_detected", False) and factors["f_phone"] > 0.35:
            active_reasons.append("Don't use the phone — Keep your eyes on the road")
        if phone_data.get("bottle_detected", False):
            active_reasons.append("Drinking / Bottle Near Face While Driving")
        if emotion_data and emotion_data.get("is_stressed", False):
            active_reasons.append(f"Elevated Stress ({emotion_data.get('dominant_emotion')})")

        return {
            "score": round(fused_score, 3),
            "math_score": round(math_score, 3),
            "level": level,
            "status_text": status_text,
            "color_hex": color_hex,
            "factors": factors,
            "primary_reasons": active_reasons,
            "ml_risk": ml_result,
            "ml_risk_probability": ml_result["confidence"],
            "ml_risk_level": ml_result["predicted_class"]
        }
