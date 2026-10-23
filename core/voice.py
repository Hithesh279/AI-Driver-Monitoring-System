"""
Voice Alerts.

Uses pyttsx3 — a fully offline, pretrained TTS engine (wraps the OS's own
speech engine: SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on
Linux). No model training, no API key, no internet needed.

IMPORTANT: pyttsx3 has a well-known bug where reusing ONE engine instance
across multiple say()+runAndWait() calls from a background thread often
speaks only the very first message, then silently stops responding
(especially the Windows SAPI5 driver, and often on Linux/espeak too when
called repeatedly from a non-main thread). The reliable fix — used here —
is to create a brand-new pyttsx3 engine for every single message and
dispose of it right after. This is what makes repeated warnings actually
keep firing every time, not just once.
"""

import queue
import threading
import time
from typing import Optional

from config import VOICE_ALERTS_ENABLED, VOICE_RATE, VOICE_COOLDOWN_SEC


class VoiceAlerter:
    def __init__(self):
        self.enabled = VOICE_ALERTS_ENABLED
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._last_spoken_by_key: dict = {}   # per alert-type cooldown, not global
        self._worker = None
        self._pyttsx3_ok = False

        if self.enabled:
            try:
                import pyttsx3
                _ = pyttsx3
                self._pyttsx3_ok = True
                self._worker = threading.Thread(target=self._run, daemon=True)
                self._worker.start()
            except Exception as e:
                print(f"[VoiceAlerter] pyttsx3 unavailable ({e}); voice alerts disabled.")
                self.enabled = False

    def _run(self):
        # Windows COM needs per-thread initialization for SAPI5 to work
        # reliably when pyttsx3 is driven from a non-main thread.
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass  # not on Windows, or pywin32 not installed — fine, ignore

        import pyttsx3
        while True:
            text = self._queue.get()
            try:
                # Fresh engine every time — avoids the "speaks once then goes
                # silent" bug that comes from reusing one engine instance.
                engine = pyttsx3.init()
                engine.setProperty("rate", VOICE_RATE)
                engine.say(text)
                engine.runAndWait()
                try:
                    engine.stop()
                except Exception:
                    pass
                del engine
            except Exception as e:
                print(f"[VoiceAlerter] speak failed: {e}")

    def speak(self, text: str, key: Optional[str] = None, cooldown: Optional[float] = None):
        """Queues a message to be spoken. Cooldown is tracked per `key` (defaults
        to the text itself), so nagging about a phone doesn't block a later,
        separate nag about drowsiness — and calling this every frame while a
        condition stays active repeats the warning every `cooldown` seconds
        until it's cleared, not just once."""
        if not self.enabled:
            return
        key = key or text
        now = time.time()
        gap = VOICE_COOLDOWN_SEC if cooldown is None else cooldown
        last = self._last_spoken_by_key.get(key, 0.0)
        if now - last < gap:
            return
        self._last_spoken_by_key[key] = now
        self._queue.put(text)


ALERT_PHRASES = {
    # Specific Directional Distractions
    "DISTRACTION_LOOKING_LEFT": "Driver looking left. Face forward and focus on the road.",
    "DISTRACTION_LOOKING_RIGHT": "Driver looking right. Keep your eyes forward on the road.",
    "DISTRACTION_LOOKING_DOWN": "Look up! Do not look down while driving. Keep your eyes on the road.",
    "DISTRACTION_LOOKING_UP": "Eyes on the road. Keep your head level and watch ahead.",
    "DISTRACTION_HEAD_TILT": "Head tilt detected. Please sit upright and focus on driving.",
    "DISTRACTION_GAZE": "Eyes off the road. Focus your attention forward.",
    "OFF_ROAD_DISTRACTION": "Distraction detected. Keep your eyes and attention on the road.",

    # Handheld Object Distractions
    "PHONE_USAGE": "Put the phone away! Using a mobile phone while driving is dangerous.",
    "BOTTLE_USAGE": "Do not drink while driving. Stop the vehicle safely before drinking.",

    # Physiological Fatigue & Sleep
    "MICROSLEEP_DROWSINESS": "Wake up! Eyes closed! Wake up immediately! Pull over safely, now.",
    "DROWSY_DROOPING_EYES": "Warning. Your eyes are drooping. Stay alert!",
    "YAWNING_FATIGUE": "Yawning detected. You are getting tired. Consider taking a rest break.",
    "FATIGUE_WARNING": "High fatigue detected. Please plan a rest stop soon.",

    # Psychological State
    "HIGH_STRESS": "High stress detected. Take a deep breath and stay calm.",

    # Emergency Services
    "EMERGENCY_SOS": "Emergency SOS activated. Transmitting vehicle location and speed to safety contacts.",
}
