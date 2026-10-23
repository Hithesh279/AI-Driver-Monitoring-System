"""
MediaPipe Face Mesh Landmark Extractor with 478-point refinement (including Irises).
Includes safe fallback handling if MediaPipe is not installed in the current environment.
"""

import warnings

# Suppress Protobuf and MediaPipe internal deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import urllib.request
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, List, Tuple

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    mp = None
    HAS_MEDIAPIPE = False

from config import (
    REFINE_LANDMARKS,
    MAX_NUM_FACES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)


class FaceLandmarkDetector:
    def __init__(self):
        self.available = False
        self.use_tasks = False
        self.face_mesh = None
        self.face_landmarker = None

        if not HAS_MEDIAPIPE:
            print("[FaceLandmarkDetector] Warning: mediapipe package not found. Landmark extraction disabled.")
            return

        mp_solutions = getattr(mp, "solutions", None)
        if mp_solutions is not None and hasattr(mp_solutions, "face_mesh"):
            self.use_tasks = False
            self.mp_face_mesh = mp_solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=MAX_NUM_FACES,
                refine_landmarks=REFINE_LANDMARKS,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
            self.available = True
        else:
            try:
                self.use_tasks = True
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

                project_root = Path(__file__).resolve().parent.parent
                model_dir = project_root / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "face_landmarker.task"

                root_model = project_root / "face_landmarker.task"
                if root_model.exists() and not model_path.exists():
                    model_path = root_model
                elif not model_path.exists():
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                    print("[FaceLandmarkDetector] Downloading face landmarker task model...")
                    urllib.request.urlretrieve(url, str(model_path))

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=RunningMode.IMAGE,
                    num_faces=MAX_NUM_FACES,
                    min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
                    min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False
                )
                self.face_landmarker = FaceLandmarker.create_from_options(options)
                self.available = True
            except Exception as e:
                print(f"[FaceLandmarkDetector] Warning: Failed to initialize MediaPipe FaceLandmarker ({e}).")
                self.available = False

    def process_frame(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Extracts 2D pixel coordinates for all 478 landmarks.
        Returns None if no face is detected or MediaPipe is unavailable.
        """
        if not self.available or frame is None:
            return None

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self.use_tasks:
            rgb_frame.flags.writeable = False
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None
            landmarks_2d = []
            face_landmarks = results.multi_face_landmarks[0]
            for lm in face_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_2d.append((cx, cy))
            return landmarks_2d
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.face_landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None
            landmarks_2d = []
            face_landmarks = results.face_landmarks[0]
            for lm in face_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_2d.append((cx, cy))
            return landmarks_2d

    def get_landmarks_3d(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts normalized 3D coordinates (x, y, z) for head pose estimation.
        """
        if not self.available or frame is None:
            return None

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self.use_tasks:
            rgb_frame.flags.writeable = False
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_3d = []
            for lm in face_landmarks.landmark:
                landmarks_3d.append([lm.x * w, lm.y * h, lm.z * w])
            return np.array(landmarks_3d, dtype=np.float64)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.face_landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None
            face_landmarks = results.face_landmarks[0]
            landmarks_3d = []
            for lm in face_landmarks:
                landmarks_3d.append([lm.x * w, lm.y * h, lm.z * w])
            return np.array(landmarks_3d, dtype=np.float64)

    def close(self):
        try:
            if not self.use_tasks and self.face_mesh is not None:
                self.face_mesh.close()
            elif self.use_tasks and self.face_landmarker is not None:
                self.face_landmarker.close()
        except Exception:
            pass
