"""
Google Maps Platform Integration Service.
Provides:
  - Real-time GPS vehicle positioning, heading, and speed telemetry.
  - GPS breadcrumb trail history for route tracing.
  - Emergency Safe Haven Finder: Nearest Rest Areas, Fuel Stations, Coffee Shops,
    and Hospitals with 1-click Google Maps Navigation directions.
  - Google Maps Static API & Direction Links for incident reports and email alerts.
"""

import time
import math
import requests
import threading
from typing import Dict, Any, List, Optional
from collections import deque

try:
    from config import (
        GOOGLE_MAPS_API_KEY,
        GOOGLE_MAPS_MAP_ID,
        GOOGLE_MAPS_ZOOM,
        MAPS_BREADCRUMB_LIMIT,
        SAFE_HAVEN_SEARCH_RADIUS_KM
    )
except ImportError:
    from ..config import (
        GOOGLE_MAPS_API_KEY,
        GOOGLE_MAPS_MAP_ID,
        GOOGLE_MAPS_ZOOM,
        MAPS_BREADCRUMB_LIMIT,
        SAFE_HAVEN_SEARCH_RADIUS_KM
    )


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes the great-circle distance between two GPS points in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(R * c)


class GoogleMapsService:
    """
    Central Google Maps integration manager for GuardianDrive AI.
    """

    def __init__(self, gps_tracker=None):
        self.gps_tracker = gps_tracker
        self.api_key = GOOGLE_MAPS_API_KEY
        self.map_id = GOOGLE_MAPS_MAP_ID
        self.zoom = GOOGLE_MAPS_ZOOM
        
        self.breadcrumbs = deque(maxlen=MAPS_BREADCRUMB_LIMIT)
        self.lock = threading.Lock()
        
        self.current_speed_kmh = 0.0
        self.current_heading_deg = 0.0
        self.is_stationary = True
        self.last_update_time = time.time()
        self.last_lat = None
        self.last_lon = None
        self._cached_havens: List[Dict[str, Any]] = []
        self._last_haven_lat = 0.0
        self._last_haven_lon = 0.0
        self._last_haven_time = 0.0
        self._is_fetching_havens = False

    def update_position(self, lat: float, lon: float,
                        reported_speed_kmh: Optional[float] = None,
                        reported_heading: Optional[float] = None) -> Dict[str, Any]:
        """
        Updates the vehicle's position, heading, and speed based on genuine live data.
        Never generates fake speeds. If the vehicle/device is not moving, speed is strictly 0.0 km/h.
        """
        now = time.time()
        dt = max(0.001, now - self.last_update_time)
        
        with self.lock:
            if reported_speed_kmh is not None:
                # Direct live speed from device GPS
                self.current_speed_kmh = max(0.0, round(float(reported_speed_kmh), 1))
                self.is_stationary = self.current_speed_kmh < 1.0
            elif self.last_lat is not None and self.last_lon is not None:
                dist_km = haversine_distance_km(self.last_lat, self.last_lon, lat, lon)
                dist_meters = dist_km * 1000.0

                # GPS sensor jitter filter: shifts under 3 meters while stationary are sensor noise
                if dist_meters < 3.0:
                    self.current_speed_kmh = 0.0
                    self.is_stationary = True
                else:
                    speed = dist_km / (dt / 3600.0)
                    if 1.0 <= speed <= 220.0:
                        self.current_speed_kmh = round(speed, 1)
                        self.is_stationary = False
                    else:
                        self.current_speed_kmh = 0.0
                        self.is_stationary = True

            if reported_heading is not None and reported_heading > 0:
                self.current_heading_deg = round(float(reported_heading), 1)
            elif self.last_lat is not None and self.last_lon is not None and not self.is_stationary:
                # Calculate bearing only when actively moving
                d_lon = math.radians(lon - self.last_lon)
                y = math.sin(d_lon) * math.cos(math.radians(lat))
                x = (math.cos(math.radians(self.last_lat)) * math.sin(math.radians(lat)) -
                     math.sin(math.radians(self.last_lat)) * math.cos(math.radians(lat)) * math.cos(d_lon))
                bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
                self.current_heading_deg = round(bearing, 1)

            # Record breadcrumb only on real displacement (>= 5 meters) or first point
            pt = (round(lat, 6), round(lon, 6))
            if not self.breadcrumbs:
                self.breadcrumbs.append(pt)
            elif not self.is_stationary:
                last_pt = self.breadcrumbs[-1]
                if haversine_distance_km(last_pt[0], last_pt[1], lat, lon) * 1000.0 >= 5.0:
                    self.breadcrumbs.append(pt)

            self.last_lat = lat
            self.last_lon = lon
            self.last_update_time = now

        return {
            "lat": lat,
            "lon": lon,
            "speed_kmh": round(self.current_speed_kmh, 1),
            "is_stationary": self.is_stationary,
            "heading_deg": self.current_heading_deg,
            "breadcrumbs_count": len(self.breadcrumbs),
        }

    def get_breadcrumbs(self) -> List[List[float]]:
        with self.lock:
            return [[pt[0], pt[1]] for pt in self.breadcrumbs]

    def get_google_maps_config(self) -> Dict[str, Any]:
        """Provides frontend configuration for Google Maps initialization."""
        loc = self.gps_tracker.get_location() if self.gps_tracker else {"lat": 14.4673, "lon": 78.8242}
        return {
            "has_api_key": bool(self.api_key),
            "api_key": self.api_key,
            "map_id": self.map_id,
            "zoom": self.zoom,
            "center": loc,
            "breadcrumbs": self.get_breadcrumbs(),
            "speed_kmh": round(self.current_speed_kmh, 1),
            "heading_deg": self.current_heading_deg,
        }

    def _fetch_live_safe_havens(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """
        Queries real, authentic physical places (Hospitals, Fuel Stations, Rest Stops, Coffee)
        around the driver's exact coordinates using Google Places API (if API key provided)
        or OpenStreetMap Nominatim Live Search.
        """
        results: List[Dict[str, Any]] = []

        # 1. Try Google Places API (New) and legacy Nearby Search if Google API key is configured
        if self.api_key:
            # 1a. Preferred: Modern Places API (New)
            try:
                new_places_url = "https://places.googleapis.com/v1/places:searchNearby"
                headers = {
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": self.api_key,
                    "X-Goog-FieldMask": "places.id,places.displayName,places.primaryType,places.types,places.location,places.formattedAddress,places.shortFormattedAddress",
                    "X-Goog-Maps-Solution-ID": "gmp_git_agentskills_v1",
                }
                body = {
                    "includedTypes": ["hospital", "gas_station", "cafe", "rest_stop"],
                    "maxResultCount": 8,
                    "locationRestriction": {
                        "circle": {
                            "center": {"latitude": lat, "longitude": lon},
                            "radius": float(SAFE_HAVEN_SEARCH_RADIUS_KM * 1000.0)
                        }
                    }
                }
                resp = requests.post(new_places_url, json=body, headers=headers, timeout=3.5)
                if resp.status_code == 200:
                    data = resp.json()
                    for p in data.get("places", [])[:6]:
                        loc_obj = p.get("location", {})
                        if not loc_obj or "latitude" not in loc_obj or "longitude" not in loc_obj:
                            continue
                        p_lat = float(loc_obj["latitude"])
                        p_lon = float(loc_obj["longitude"])
                        p_name = p.get("displayName", {}).get("text", "Nearby Safe Haven")
                        dist_km = round(haversine_distance_km(lat, lon, p_lat, p_lon), 1)
                        eta_mins = max(2, int(round((dist_km / 50.0) * 60)))
                        types = p.get("types", []) or [p.get("primaryType", "")]
                        cat = "Emergency Care" if any("hospital" in t for t in types) else (
                            "Fuel & Service" if any("gas_station" in t for t in types) else "Rest & Cafe"
                        )
                        icon = "🏥" if "Emergency Care" in cat else ("⛽" if "Fuel" in cat else "☕")
                        results.append({
                            "id": p.get("id", f"place_{len(results)}"),
                            "name": p_name,
                            "category": cat,
                            "icon": icon,
                            "lat": round(p_lat, 6),
                            "lon": round(p_lon, 6),
                            "distance_km": dist_km,
                            "eta_minutes": eta_mins,
                            "amenities": [p.get("shortFormattedAddress") or p.get("formattedAddress", "Verified Live Location")],
                            "google_maps_nav_url": self.get_google_maps_directions_url(lat, lon, p_lat, p_lon),
                            "google_maps_view_url": f"https://www.google.com/maps/place/?q=place_id:{p.get('id')}",
                        })
            except Exception:
                pass

            # 1b. Fallback: Classic Nearby Search
            if not results:
                try:
                    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                    params = {
                        "location": f"{lat:.6f},{lon:.6f}",
                        "radius": int(SAFE_HAVEN_SEARCH_RADIUS_KM * 1000),
                        "type": "hospital|gas_station|cafe",
                        "key": self.api_key,
                        "solution_id": "gmp_git_agentskills_v1",
                    }
                    resp = requests.get(places_url, params=params, timeout=3.5)
                    if resp.status_code == 200:
                        data = resp.json()
                        for p in data.get("results", [])[:6]:
                            p_lat = float(p["geometry"]["location"]["lat"])
                            p_lon = float(p["geometry"]["location"]["lng"])
                            dist_km = round(haversine_distance_km(lat, lon, p_lat, p_lon), 1)
                            eta_mins = max(2, int(round((dist_km / 50.0) * 60)))
                            types = p.get("types", [])
                            cat = "Emergency Care" if "hospital" in types else ("Fuel & Service" if "gas_station" in types else "Rest & Cafe")
                            icon = "🏥" if "hospital" in types else ("⛽" if "gas_station" in types else "☕")
                            results.append({
                                "id": p.get("place_id", f"place_{len(results)}"),
                                "name": p.get("name", "Nearby Safe Haven"),
                                "category": cat,
                                "icon": icon,
                                "lat": round(p_lat, 6),
                                "lon": round(p_lon, 6),
                                "distance_km": dist_km,
                                "eta_minutes": eta_mins,
                                "amenities": [p.get("vicinity", "Verified Live Location")],
                                "google_maps_nav_url": self.get_google_maps_directions_url(lat, lon, p_lat, p_lon),
                                "google_maps_view_url": f"https://www.google.com/maps/place/?q=place_id:{p.get('place_id')}",
                            })
                except Exception:
                    pass

        # 2. If no results yet, query OpenStreetMap Nominatim Live Search
        if not results:
            headers = {"User-Agent": "GuardianDriveDMS/1.0 (contact@guardiandrive.ai)"}
            d = 0.08
            viewbox = f"{lon-d:.5f},{lat+d:.5f},{lon+d:.5f},{lat-d:.5f}"
            categories = [
                ("hospital", "Medical Emergency", "🏥", ["24/7 Emergency", "Ambulance", "Trauma Care"]),
                ("fuel", "Fuel & Energy", "⛽", ["Fuel Station", "Air/Water", "Convenience"]),
                ("cafe", "Coffee & Rest", "☕", ["Hot Beverages", "Refreshment", "Rest Bay"]),
            ]
            for query_term, cat_name, cat_icon, default_amenities in categories:
                try:
                    url = f"https://nominatim.openstreetmap.org/search?format=json&q={query_term}&bounded=1&viewbox={viewbox}&limit=2"
                    r = requests.get(url, headers=headers, timeout=3.0)
                    if r.status_code == 200:
                        places = r.json()
                        for p in places:
                            p_lat = float(p["lat"])
                            p_lon = float(p["lon"])
                            p_name = p.get("display_name", "").split(",")[0].strip() or f"Nearby {cat_name}"
                            dist_km = round(haversine_distance_km(lat, lon, p_lat, p_lon), 1)
                            eta_mins = max(2, int(round((dist_km / 50.0) * 60)))
                            results.append({
                                "id": f"osm_{p.get('osm_id', len(results))}",
                                "name": p_name,
                                "category": cat_name,
                                "icon": cat_icon,
                                "lat": round(p_lat, 6),
                                "lon": round(p_lon, 6),
                                "distance_km": dist_km,
                                "eta_minutes": eta_mins,
                                "amenities": default_amenities,
                                "google_maps_nav_url": self.get_google_maps_directions_url(lat, lon, p_lat, p_lon),
                                "google_maps_view_url": f"https://www.google.com/maps/search/?api=1&query={p_lat:.6f},{p_lon:.6f}",
                            })
                except Exception:
                    continue

        if not results:
            results = self._get_default_intent_havens(lat, lon)

        results.sort(key=lambda x: x["distance_km"])
        return results

    def _get_default_intent_havens(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """Instant fail-safe fallback: generates immediate Google Maps POI search links."""
        categories_intent = [
            ("Live Hospital Search", "Medical Emergency", "🏥", "hospital", ["24/7 Emergency", "Trauma Care"]),
            ("Live Fuel Station Search", "Fuel & Energy", "⛽", "fuel+station", ["Fuel Station", "Rest Stop"]),
            ("Live Coffee & Rest Area", "Coffee & Rest", "☕", "coffee+rest+area", ["Caffeine", "Rest Area"]),
        ]
        results = []
        for name, cat, icon, query, amenities in categories_intent:
            nav_url = f"https://www.google.com/maps/search/{query}/@{lat:.6f},{lon:.6f},14z"
            results.append({
                "id": f"intent_{query}",
                "name": name,
                "category": cat,
                "icon": icon,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "distance_km": 0.0,
                "eta_minutes": 2,
                "amenities": amenities,
                "google_maps_nav_url": nav_url,
                "google_maps_view_url": nav_url,
            })
        return results

    def _async_fetch_safe_havens(self, lat: float, lon: float):
        """Background worker thread target that performs network requests without blocking the video pipeline."""
        try:
            fresh = self._fetch_live_safe_havens(lat, lon)
            if fresh:
                with self.lock:
                    self._cached_havens = fresh
                    self._last_haven_lat = lat
                    self._last_haven_lon = lon
                    self._last_haven_time = time.time()
        except Exception:
            pass
        finally:
            with self.lock:
                self._is_fetching_havens = False

    def find_nearby_safe_havens(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """
        Returns cached or fresh live safe havens around the driver's authentic location.
        Non-blocking: immediately returns cached havens (or instant Google Maps links)
        and triggers a background thread to refresh without freezing the camera frame loop.
        """
        now = time.time()
        with self.lock:
            if not self._cached_havens:
                self._cached_havens = self._get_default_intent_havens(lat, lon)
                self._last_haven_lat = lat
                self._last_haven_lon = lon
                self._last_haven_time = 0.0  # Mark stale so background worker triggers immediately

            is_stale = (
                (now - self._last_haven_time >= 60.0) or
                abs(lat - self._last_haven_lat) >= 0.005 or
                abs(lon - self._last_haven_lon) >= 0.005
            )

            if is_stale and not self._is_fetching_havens:
                self._is_fetching_havens = True
                t = threading.Thread(
                    target=self._async_fetch_safe_havens,
                    args=(lat, lon),
                    daemon=True
                )
                t.start()

            return list(self._cached_havens)

    def get_google_maps_directions_url(self, origin_lat: float, origin_lon: float,
                                       dest_lat: float, dest_lon: float) -> str:
        """Returns standard Google Maps driving directions deep-link."""
        return (f"https://www.google.com/maps/dir/?api=1"
                f"&origin={origin_lat:.6f},{origin_lon:.6f}"
                f"&destination={dest_lat:.6f},{dest_lon:.6f}"
                f"&travelmode=driving")

    def get_static_map_preview_url(self, lat: float, lon: float, zoom: int = 15, width: int = 600, height: int = 300) -> str:
        """
        Generates Google Maps Static API URL if API key is configured,
        or a universal Google Maps location link fallback.
        """
        if self.api_key:
            return (
                f"https://maps.googleapis.com/maps/api/staticmap?"
                f"center={lat},{lon}&zoom={zoom}&size={width}x{height}"
                f"&markers=color:red%7Clabel:V%7C{lat},{lon}&key={self.api_key}"
            )
        return f"https://www.google.com/maps?q={lat},{lon}"
