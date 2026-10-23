"""
Thread-safe SQLite Database Manager for telemetry and incident logging.
"""

import sqlite3
import threading
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from config import DB_PATH


class DatabaseManager:
    _instances: Dict[str, "DatabaseManager"] = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Union[Path, str]] = None):
        resolved_path = str(Path(db_path or DB_PATH).resolve())
        with cls._lock:
            if resolved_path not in cls._instances:
                instance = super(DatabaseManager, cls).__new__(cls)
                instance._init_db(Path(resolved_path))
                cls._instances[resolved_path] = instance
            return cls._instances[resolved_path]

    def _init_db(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self.local = threading.local()
        self.instance_lock = threading.RLock()
        with self.instance_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use DELETE journal mode for cross-platform compatibility (Docker mounts, network drives, Windows/Linux)
                try:
                    cursor.execute("PRAGMA journal_mode=DELETE;")
                except sqlite3.OperationalError:
                    pass

                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time REAL,
                        end_time REAL,
                        duration_sec REAL DEFAULT 0,
                        avg_risk REAL DEFAULT 0,
                        max_risk REAL DEFAULT 0,
                        total_alerts INTEGER DEFAULT 0,
                        total_yawns INTEGER DEFAULT 0,
                        phone_events INTEGER DEFAULT 0
                    )
                """)

                # Trips Table (driver, passenger, emergency contacts)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trips (
                        trip_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        driver_name TEXT,
                        driver_email TEXT,
                        passenger_name TEXT,
                        passenger_emergency_email TEXT,
                        vehicle_id TEXT,
                        start_time REAL,
                        end_time REAL,
                        status TEXT DEFAULT 'ACTIVE'
                    )
                """)

                # Telemetry Logs (sampled periodically)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp REAL,
                        ear REAL,
                        mar REAL,
                        perclos REAL,
                        yaw REAL,
                        pitch REAL,
                        roll REAL,
                        phone_detected INTEGER,
                        risk_score REAL,
                        risk_level TEXT,
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    )
                """)

                # Alert Incidents Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp REAL,
                        alert_type TEXT,
                        severity TEXT,
                        trigger_value REAL,
                        details TEXT,
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    )
                """)
                conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self.local, "conn") or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn

    def start_session(self, session_id: str, start_time: Optional[float] = None) -> None:
        start_time = start_time or time.time()
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (
                    session_id, start_time, end_time, duration_sec,
                    avg_risk, max_risk, total_alerts, total_yawns, phone_events
                )
                VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0)
            """, (session_id, start_time, start_time))
            conn.commit()

    def start_trip(self, trip_id: str, session_id: str, driver_name: str, driver_email: str,
                   passenger_name: str, passenger_emergency_email: str, vehicle_id: str) -> None:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trips (
                    trip_id, session_id, driver_name, driver_email,
                    passenger_name, passenger_emergency_email, vehicle_id, start_time, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
            """, (trip_id, session_id, driver_name, driver_email,
                  passenger_name, passenger_emergency_email, vehicle_id, time.time()))
            conn.commit()

    def get_trip(self, trip_id: str) -> Optional[Dict[str, Any]]:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trips WHERE trip_id = ? OR session_id = ? ORDER BY start_time DESC LIMIT 1",
                           (trip_id, trip_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_latest_trip(self) -> Optional[Dict[str, Any]]:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trips ORDER BY start_time DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def log_telemetry(self, session_id: str, timestamp: float, ear: float, mar: float,
                      perclos: float, yaw: float, pitch: float, roll: float,
                      phone_detected: bool, risk_score: float, risk_level: str,
                      total_yawns: Optional[int] = None) -> None:
        try:
            with self.instance_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO telemetry_logs (
                        session_id, timestamp, ear, mar, perclos, yaw, pitch, roll,
                        phone_detected, risk_score, risk_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, timestamp, ear, mar, perclos, yaw, pitch, roll,
                      1 if phone_detected else 0, risk_score, risk_level))

                # Keep session summary statistics updated in real time
                cursor.execute("""
                    SELECT
                        AVG(risk_score) as avg_risk,
                        MAX(risk_score) as max_risk,
                        SUM(phone_detected) as phone_count
                    FROM telemetry_logs
                    WHERE session_id = ?
                """, (session_id,))
                stats = cursor.fetchone()
                avg_r = float(stats["avg_risk"] or 0.0) if stats else float(risk_score)
                max_r = float(stats["max_risk"] or 0.0) if stats else float(risk_score)
                phone_cnt = int(stats["phone_count"] or 0) if stats else (1 if phone_detected else 0)

                if total_yawns is not None:
                    cursor.execute("""
                        UPDATE sessions
                        SET end_time = ?,
                            duration_sec = MAX(0.0, ? - start_time),
                            avg_risk = ?,
                            max_risk = MAX(max_risk, ?),
                            phone_events = ?,
                            total_yawns = MAX(total_yawns, ?)
                        WHERE session_id = ?
                    """, (timestamp, timestamp, avg_r, max_r, phone_cnt, int(total_yawns), session_id))
                else:
                    cursor.execute("""
                        UPDATE sessions
                        SET end_time = ?,
                            duration_sec = MAX(0.0, ? - start_time),
                            avg_risk = ?,
                            max_risk = MAX(max_risk, ?),
                            phone_events = ?
                        WHERE session_id = ?
                    """, (timestamp, timestamp, avg_r, max_r, phone_cnt, session_id))
                conn.commit()
        except Exception:
            pass

    def log_alert(self, session_id: str, timestamp: float, alert_type: str,
                  severity: str, trigger_value: float, details: str = "") -> None:
        try:
            with self.instance_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alerts (session_id, timestamp, alert_type, severity, trigger_value, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, timestamp, alert_type, severity, trigger_value, details))

                # Increment total alerts (and yawn/phone counters if applicable) in session
                yawn_inc = 1 if alert_type == "YAWNING_FATIGUE" else 0
                phone_inc = 1 if alert_type == "PHONE_USAGE" else 0
                cursor.execute("""
                    UPDATE sessions
                    SET total_alerts = total_alerts + 1,
                        total_yawns = total_yawns + ?,
                        phone_events = MAX(phone_events, phone_events + ?),
                        end_time = MAX(end_time, ?),
                        duration_sec = MAX(duration_sec, ? - start_time)
                    WHERE session_id = ?
                """, (yawn_inc, phone_inc, timestamp, timestamp, session_id))
                conn.commit()
        except Exception:
            pass

    def end_session(self, session_id: str, end_time: Optional[float] = None) -> Dict[str, Any]:
        end_time = end_time or time.time()
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Calculate summary stats from telemetry
            cursor.execute("""
                SELECT
                    COUNT(*) as count,
                    AVG(risk_score) as avg_risk,
                    MAX(risk_score) as max_risk,
                    SUM(phone_detected) as phone_count
                FROM telemetry_logs
                WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()

            avg_risk = float(row["avg_risk"]) if row and row["avg_risk"] is not None else 0.0
            max_risk = float(row["max_risk"]) if row and row["max_risk"] is not None else 0.0
            phone_count = int(row["phone_count"]) if row and row["phone_count"] is not None else 0

            cursor.execute("SELECT start_time FROM sessions WHERE session_id = ?", (session_id,))
            s_row = cursor.fetchone()
            start_time = s_row["start_time"] if s_row else end_time
            duration = max(0.0, end_time - start_time)

            cursor.execute("""
                UPDATE sessions
                SET end_time = ?, duration_sec = ?, avg_risk = ?, max_risk = ?, phone_events = MAX(phone_events, ?)
                WHERE session_id = ?
            """, (end_time, duration, avg_risk, max_risk, phone_count, session_id))

            cursor.execute("""
                UPDATE trips
                SET end_time = ?, status = 'COMPLETED'
                WHERE session_id = ? AND status = 'ACTIVE'
            """, (end_time, session_id))
            conn.commit()

            return {
                "session_id": session_id,
                "duration_sec": duration,
                "avg_risk": avg_risk,
                "max_risk": max_risk,
                "phone_events": phone_count
            }

    def get_recent_telemetry(self, session_id: str, limit: int = 60) -> List[Dict[str, Any]]:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM telemetry_logs
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]

    def get_recent_alerts(self, session_id: str, limit: int = 15) -> List[Dict[str, Any]]:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC LIMIT 50")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_telemetry_in_timerange(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM telemetry_logs
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (start_time, end_time))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_alerts_in_timerange(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """, (start_time, end_time))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_sessions_in_timerange(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        with self.instance_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions
                WHERE start_time >= ? AND start_time <= ?
                ORDER BY start_time DESC
            """, (start_time, end_time))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_timeframe_data(self, hours: float = 24.0) -> Dict[str, Any]:
        """Fetches telemetry, alerts, and sessions for the specified lookback window."""
        now = time.time()
        start_time = now - (hours * 3600.0)
        telemetry = self.get_telemetry_in_timerange(start_time, now)
        alerts = self.get_alerts_in_timerange(start_time, now)
        sessions = self.get_sessions_in_timerange(start_time, now)
        return {
            "start_time": start_time,
            "end_time": now,
            "hours": hours,
            "telemetry": telemetry,
            "alerts": alerts,
            "sessions": sessions
        }
