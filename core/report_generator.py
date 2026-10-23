"""
Automated Driving Performance Report Generator (PPT Module 05).
Generates Daily, Weekly, and Monthly driving summaries, driver safety scores (0-100),
incident breakdowns, and personalized coaching recommendations.
"""

import time
import html
from typing import Dict, Any
from database.db_manager import DatabaseManager

DISTRACTION_ALERT_TYPES = {
    "OFF_ROAD_DISTRACTION",
    "DISTRACTION_LOOKING_DOWN",
    "DISTRACTION_LOOKING_LEFT",
    "DISTRACTION_LOOKING_RIGHT",
    "DISTRACTION_LOOKING_UP",
    "DISTRACTION_HEAD_TILT",
    "DISTRACTION_GAZE",
}

FATIGUE_ALERT_TYPES = {
    "FATIGUE_WARNING",
    "DROWSY_DROOPING_EYES",
    "YAWNING_FATIGUE",
}


class ReportGenerator:
    """
    Analyzes historical telemetry and alert logs from the database
    to produce structured daily, weekly, and monthly driving summaries.
    """

    def __init__(self, db: DatabaseManager = None):
        self.db = db or DatabaseManager()

    def generate_report(self, timeframe: str = "daily") -> Dict[str, Any]:
        """
        Generates a driving report for the given timeframe:
          - 'daily': past 24 hours
          - 'weekly': past 7 days
          - 'monthly': past 30 days
        """
        hours_map = {
            "daily": 24.0,
            "weekly": 168.0,
            "monthly": 720.0
        }
        hours = hours_map.get(timeframe.lower(), 24.0)
        data = self.db.get_timeframe_data(hours=hours)

        telemetry = data["telemetry"]
        alerts = data["alerts"]
        sessions = data["sessions"]

        # 1. Driving Duration & Session Counts
        total_sessions = len(sessions)
        total_duration_sec = sum(s.get("duration_sec", 0.0) or 0.0 for s in sessions)
        if total_duration_sec == 0 and telemetry:
            # Estimate from telemetry span
            total_duration_sec = max(60.0, float(len(telemetry)))
        
        hours_driven = round(total_duration_sec / 3600.0, 2)
        minutes_driven = int(round(total_duration_sec / 60.0))

        # 2. Alert & Incident Breakdown (rolls up granular directional & fatigue alerts into primary categories)
        alert_counts: Dict[str, int] = {
            "MICROSLEEP_DROWSINESS": 0,
            "PHONE_USAGE": 0,
            "OFF_ROAD_DISTRACTION": 0,
            "BOTTLE_USAGE": 0,
            "FATIGUE_WARNING": 0,
            "HIGH_STRESS": 0,
            "EMERGENCY_SOS": 0,
        }
        detailed_breakdown: Dict[str, int] = {}

        for a in alerts:
            atype = a.get("alert_type", "UNKNOWN")
            detailed_breakdown[atype] = detailed_breakdown.get(atype, 0) + 1

            if atype in DISTRACTION_ALERT_TYPES:
                alert_counts["OFF_ROAD_DISTRACTION"] += 1
            elif atype in FATIGUE_ALERT_TYPES:
                alert_counts["FATIGUE_WARNING"] += 1
            elif atype in alert_counts:
                alert_counts[atype] += 1
            else:
                alert_counts[atype] = 1

        total_alerts = len(alerts)

        # 3. Telemetry Averages
        if telemetry:
            avg_ear = sum(t.get("ear", 0.0) for t in telemetry) / len(telemetry)
            avg_mar = sum(t.get("mar", 0.0) for t in telemetry) / len(telemetry)
            avg_perclos = sum(t.get("perclos", 0.0) for t in telemetry) / len(telemetry)
            avg_risk = sum(t.get("risk_score", 0.0) for t in telemetry) / len(telemetry)
            max_risk = max((t.get("risk_score", 0.0) for t in telemetry), default=0.0)
        else:
            avg_ear = 0.30
            avg_mar = 0.18
            avg_perclos = 0.03
            avg_risk = 0.15
            max_risk = 0.25

        # 4. Overall Driving Safety Score Calculation (0 to 100 Index)
        # 100 base score with weighted deductions for hazardous behaviors
        penalty_microsleep = alert_counts["MICROSLEEP_DROWSINESS"] * 15
        penalty_phone = alert_counts["PHONE_USAGE"] * 10
        penalty_distraction = alert_counts["OFF_ROAD_DISTRACTION"] * 6
        penalty_bottle = alert_counts["BOTTLE_USAGE"] * 5
        penalty_fatigue = alert_counts["FATIGUE_WARNING"] * 4
        penalty_stress = alert_counts["HIGH_STRESS"] * 3
        penalty_risk = int(avg_risk * 25.0)

        total_penalty = (
            penalty_microsleep + penalty_phone + penalty_distraction +
            penalty_bottle + penalty_fatigue + penalty_stress + penalty_risk
        )
        safety_score = max(20, min(100, 100 - total_penalty))

        if safety_score >= 90:
            rating = "Excellent / Highly Attentive"
            grade = "A+"
            grade_color = "#22c55e"
        elif safety_score >= 75:
            rating = "Good / Generally Safe"
            grade = "B"
            grade_color = "#3b82f6"
        elif safety_score >= 60:
            rating = "Fair / Needs Attention"
            grade = "C"
            grade_color = "#eab308"
        else:
            rating = "Critical Risk / High Hazard"
            grade = "D"
            grade_color = "#ef4444"

        # 5. Personalized AI Coaching Recommendations (PPT Slide 11 & 17)
        recommendations = []
        if alert_counts["MICROSLEEP_DROWSINESS"] > 0:
            recommendations.append(
                "🚨 Critical eye closure events were recorded. Do not drive when sleep-deprived; "
                "schedule a mandatory 20-minute power nap before long highway trips."
            )
        if alert_counts["PHONE_USAGE"] > 0:
            recommendations.append(
                "📱 Mobile phone use was detected while the vehicle was in motion. "
                "Activate 'Do Not Disturb While Driving' or store your device in the glove compartment."
            )
        if alert_counts["BOTTLE_USAGE"] > 0:
            recommendations.append(
                "🍼 In-motion drinking/bottle usage was detected. Pull over to a safe rest area before hydrating."
            )
        if alert_counts["OFF_ROAD_DISTRACTION"] >= 2:
            recommendations.append(
                "👀 Frequent off-road head turns or downward glances observed. Keep your visual focus centered on the road "
                "and set up navigation and mirrors before departing."
            )
        if avg_perclos > 0.10 or alert_counts["FATIGUE_WARNING"] > 0:
            recommendations.append(
                "☕ Cumulative eye fatigue (PERCLOS) or yawning is elevated. Take a 15-minute hydration and walk break "
                "every 90 minutes of continuous driving."
            )
        if alert_counts["HIGH_STRESS"] > 0:
            recommendations.append(
                "🧘 Elevated driver stress detected. Practice slow, deep breathing and maintain extra following distance."
            )
        if not recommendations:
            recommendations.append(
                "🌟 Excellent driving performance! Attentive road focus and zero critical fatigue alerts. Keep up the great habits."
            )

        return {
            "timeframe": timeframe.capitalize(),
            "timeframe_key": timeframe.lower(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lookback_hours": hours,
            "safety_score": safety_score,
            "grade": grade,
            "grade_color": grade_color,
            "rating": rating,
            "total_sessions": total_sessions,
            "hours_driven": hours_driven,
            "minutes_driven": minutes_driven,
            "total_alerts": total_alerts,
            "alert_breakdown": alert_counts,
            "detailed_alert_breakdown": detailed_breakdown,
            "metrics": {
                "avg_ear": round(avg_ear, 3),
                "avg_mar": round(avg_mar, 3),
                "avg_perclos_pct": round(avg_perclos * 100.0, 1),
                "avg_risk": round(avg_risk, 3),
                "max_risk": round(max_risk, 3),
            },
            "recommendations": recommendations,
        }

    def render_html_report(self, timeframe: str = "daily") -> str:
        """Generates styled HTML report for direct in-browser inspection and printing."""
        rep = self.generate_report(timeframe)
        rec_html = "".join(f"<li style='margin-bottom:8px;'>{html.escape(r)}</li>" for r in rep["recommendations"])
        safe_timeframe = html.escape(str(rep['timeframe']))
        safe_rating = html.escape(str(rep['rating']))
        safe_grade = html.escape(str(rep['grade']))

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
          <title>Driver Safety Report — {safe_timeframe}</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #e6edf3; padding: 24px; max-width: 800px; margin: 0 auto; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
            h1, h2, h3 {{ color: #f0f6fc; margin-top: 0; }}
            .score-badge {{ font-size: 42px; font-weight: 800; color: {rep['grade_color']}; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }}
            .stat {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }}
            .stat-val {{ font-size: 22px; font-weight: 700; color: #58a6ff; margin-top: 4px; }}
            .stat-label {{ font-size: 12px; color: #8b949e; text-transform: uppercase; }}
            ul {{ padding-left: 20px; }}
          </style>
        </head>
        <body>
          <div class="card">
            <h1>🚗 GuardianDrive AI — {safe_timeframe} Safety Report</h1>
            <p style="color:#8b949e;">Generated: {rep['generated_at']} &nbsp;|&nbsp; Period: Past {rep['lookback_hours']} hours</p>
            <div style="display:flex; align-items:center; justify-content:space-between; margin-top:16px;">
              <div>
                <div class="stat-label">Driver Safety Score</div>
                <div class="score-badge">{rep['safety_score']} / 100 <span style="font-size:24px;">({safe_grade})</span></div>
                <div style="color:{rep['grade_color']}; font-weight:600;">{safe_rating}</div>
              </div>
              <div style="text-align:right;">
                <div class="stat-label">Total Time Monitored</div>
                <div style="font-size:26px; font-weight:700;">{rep['minutes_driven']} mins</div>
                <div style="color:#8b949e;">{rep['total_sessions']} Driving Session(s)</div>
              </div>
            </div>
          </div>

          <div class="card">
            <h3>Incident & Alert Summary</h3>
            <div class="grid">
              <div class="stat"><div class="stat-label">Total Alerts</div><div class="stat-val">{rep['total_alerts']}</div></div>
              <div class="stat"><div class="stat-label">Microsleeps</div><div class="stat-val" style="color:#ef4444;">{rep['alert_breakdown']['MICROSLEEP_DROWSINESS']}</div></div>
              <div class="stat"><div class="stat-label">Phone Use</div><div class="stat-val" style="color:#ef4444;">{rep['alert_breakdown']['PHONE_USAGE']}</div></div>
              <div class="stat"><div class="stat-label">Distractions</div><div class="stat-val" style="color:#eab308;">{rep['alert_breakdown']['OFF_ROAD_DISTRACTION']}</div></div>
              <div class="stat"><div class="stat-label">Fatigue / Yawns</div><div class="stat-val" style="color:#f97316;">{rep['alert_breakdown']['FATIGUE_WARNING']}</div></div>
              <div class="stat"><div class="stat-label">Avg PERCLOS</div><div class="stat-val">{rep['metrics']['avg_perclos_pct']}%</div></div>
            </div>
          </div>

          <div class="card">
            <h3>Personalized Safety Coaching & Recommendations</h3>
            <ul style="color:#e6edf3; line-height:1.6;">
              {rec_html}
            </ul>
          </div>
        </body>
        </html>
        """
