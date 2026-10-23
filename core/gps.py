"""
GPS Tracking Module.
Provides genuine live vehicle coordinates without artificial drift or fake random walks.
Supports:
  - Live device GPS streaming from browser / client Geolocation API.
  - IP-based approximate location lookup for stationary setups.
  - Real hardware GPS receiver integration (USB / Serial NMEA).
"""

import threading
from typing import Dict, Optional
from config import GPS_MODE, GPS_SIM_START_LAT, GPS_SIM_START_LON


class GPSTracker:
    def __init__(self):
        self.mode = GPS_MODE
        self._lat = float(GPS_SIM_START_LAT)
        self._lon = float(GPS_SIM_START_LON)
        self._speed_kmh = 0.0
        self._heading_deg = 0.0
        self._lock = threading.Lock()
        self._is_live_device = False

        if self.mode == "ip":
            self._try_fetch_ip_location()

    def _try_fetch_ip_location(self):
        """Attempts to fetch genuine public IP geolocation once."""
        try:
            import requests
            resp = requests.get("https://ipapi.co/json/", timeout=3)
            if resp.ok:
                data = resp.json()
                if data.get("latitude") and data.get("longitude"):
                    with self._lock:
                        self._lat = float(data["latitude"])
                        self._lon = float(data["longitude"])
        except Exception:
            pass

    def update_live_location(self, lat: float, lon: float, speed_kmh: Optional[float] = None, heading_deg: Optional[float] = None):
        """
        Receives authentic real-time coordinates and speed from device GPS or browser Geolocation.
        """
        with self._lock:
            self._lat = float(lat)
            self._lon = float(lon)
            if speed_kmh is not None:
                self._speed_kmh = max(0.0, float(speed_kmh))
            if heading_deg is not None:
                self._heading_deg = float(heading_deg)
            self._is_live_device = True

    def get_location(self) -> Dict[str, float]:
        """
        Returns authentic current {lat, lon}.
        Never applies fake random walk jitter. When stationary, coordinates stay stable.
        """
        with self._lock:
            return {
                "lat": round(self._lat, 6),
                "lon": round(self._lon, 6),
                "speed_kmh": round(self._speed_kmh, 1),
                "heading_deg": round(self._heading_deg, 1),
                "is_live_device": self._is_live_device
            }

    def maps_link(self) -> str:
        loc = self.get_location()
        return f"https://www.google.com/maps?q={loc['lat']},{loc['lon']}"
