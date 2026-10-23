"""
Safety Advisory Generator.

No LLM, no training required: this maps a detected alert condition to a
pre-written, expert-style set of safety steps. It plays the role the PPT's
"Personalized Recommendations" module described, without the agentic
GenAI layer — deterministic, explainable, and instant (no API latency).

If you later want a real pretrained model here instead of the rule table,
the natural drop-in is a small pretrained text-generation pipeline, e.g.:

    from transformers import pipeline
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    tip = generator(f"Give a short driving safety tip for: {condition}")[0]["generated_text"]

That model ships pretrained (no training step), but it adds a heavy
dependency (~300MB) — the rule table below is the default so the project
stays lightweight and fully explainable in a viva.
"""

from typing import Dict, Any

ADVISORY_TABLE: Dict[str, Dict[str, Any]] = {
    "MICROSLEEP_DROWSINESS": {
        "condition": "Microsleep / prolonged eye closure detected",
        "severity": "CRITICAL",
        "steps": [
            "Pull over safely as soon as possible — do not continue driving.",
            "Stop at the nearest safe location (fuel station, rest area, wide shoulder).",
            "Take a 15-20 minute power nap or switch drivers if someone else can drive.",
            "Drink water or caffeine and get fresh air before resuming.",
            "Do not resume the trip until you feel fully alert.",
        ],
    },
    "DISTRACTION_LOOKING_DOWN": {
        "condition": "Driver looking down at lap, phone, or console",
        "severity": "HIGH",
        "steps": [
            "Raise your head and keep your eyes level with the road.",
            "Never check text messages or screens while in motion.",
            "Place mobile devices in a secure dash mount or glove compartment.",
        ],
    },
    "DISTRACTION_LOOKING_LEFT": {
        "condition": "Driver looking left away from the road",
        "severity": "HIGH",
        "steps": [
            "Turn your head forward toward traffic immediately.",
            "Limit side mirror checks to brief 1-second glances.",
            "Do not maintain prolonged attention toward the driver-side window.",
        ],
    },
    "DISTRACTION_LOOKING_RIGHT": {
        "condition": "Driver looking right away from the road",
        "severity": "HIGH",
        "steps": [
            "Keep your primary gaze centered on the lane ahead.",
            "Avoid prolonged attention toward the passenger seat or right window.",
            "Maintain peripheral awareness while keeping face forward.",
        ],
    },
    "DISTRACTION_LOOKING_UP": {
        "condition": "Driver head tilted up away from the road",
        "severity": "MEDIUM",
        "steps": [
            "Adjust your seating posture and keep eyes level with the road.",
            "Avoid fixing gaze on the ceiling or sun visor.",
        ],
    },
    "DISTRACTION_HEAD_TILT": {
        "condition": "Driver head tilted sideways / slumping posture",
        "severity": "MEDIUM",
        "steps": [
            "Sit upright with back firmly against the driver seat.",
            "Adjust headrest to maintain proper neck support.",
            "If head tilting is caused by sleepiness, take an immediate rest break.",
        ],
    },
    "DISTRACTION_GAZE": {
        "condition": "Driver gaze wandering away from the road",
        "severity": "MEDIUM",
        "steps": [
            "Focus your visual attention on the road and oncoming vehicles.",
            "Avoid daydreaming or visual fixation on roadside objects.",
        ],
    },
    "OFF_ROAD_DISTRACTION": {
        "condition": "Sustained head/gaze distraction from the road",
        "severity": "HIGH",
        "steps": [
            "Bring your eyes and attention back to the road immediately.",
            "If you need to check something, wait until you can safely stop.",
            "Keep both hands on the wheel and avoid multitasking while driving.",
        ],
    },
    "DROWSY_DROOPING_EYES": {
        "condition": "Driver eyelids drooping / rising drowsiness detected",
        "severity": "HIGH",
        "steps": [
            "Open vehicle windows for fresh air ventilation.",
            "Increase cabin cooling to stimulate alertness.",
            "Plan a stopping point within the next 10-15 minutes.",
        ],
    },
    "YAWNING_FATIGUE": {
        "condition": "Frequent yawning / driver fatigue accumulating",
        "severity": "MEDIUM",
        "steps": [
            "Plan a rest break within the next 15-20 minutes.",
            "Take a short walk and hydrate before continuing the trip.",
            "Avoid heavy meals right before or during driving.",
        ],
    },
    "HIGH_STRESS": {
        "condition": "High driver psychological stress / agitation detected",
        "severity": "MEDIUM",
        "steps": [
            "Take slow, deep breaths to restore calm and focus.",
            "Lower in-cabin audio volume and increase ventilation.",
            "Maintain extra following distance to prevent reactive maneuvers.",
        ],
    },
    "PHONE_USAGE": {
        "condition": "Don't use the phone — Keep your eyes on the road",
        "severity": "HIGH",
        "steps": [
            "Don't use the phone. Keep your eyes on the road.",
            "Put the phone down immediately and keep both hands on the wheel.",
            "If a call or message is urgent, pull over safely before handling the phone.",
            "Never look down at screens while operating the vehicle.",
        ],
    },
    "BOTTLE_USAGE": {
        "condition": "Drinking / bottle near face detected while driving",
        "severity": "MEDIUM",
        "steps": [
            "Do not drink while the vehicle is moving.",
            "Pull over to the side of the road and stop safely first.",
            "Once stopped, drink water, then resume driving when ready.",
        ],
    },
    "FATIGUE_WARNING": {
        "condition": "Rising cumulative fatigue (high PERCLOS / frequent yawning)",
        "severity": "MEDIUM",
        "steps": [
            "Plan a rest break within the next 15-20 minutes.",
            "Open a window or increase cabin ventilation to stay alert.",
            "Avoid heavy meals right before or during long drives.",
        ],
    },
    "EMERGENCY_SOS": {
        "condition": "Manual Emergency SOS Distress Signal Triggered by Driver",
        "severity": "CRITICAL",
        "steps": [
            "Emergency SOS button was pressed manually from in-cabin dashboard.",
            "Contact driver immediately by phone or dispatch roadside emergency assistance.",
            "Live vehicle location and 1-click turn-by-turn navigation link provided below.",
            "Check local highway emergency services if driver is unreachable.",
        ],
    },
    "DEFAULT": {
        "condition": "General safe-driving reminder",
        "severity": "LOW",
        "steps": [
            "Maintain a safe following distance.",
            "Keep both hands on the wheel and stay focused on the road.",
        ],
    },
}


def get_advisory(alert_type: str, risk_level: str = "Medium") -> Dict[str, Any]:
    """Returns the driver's condition and a short, ordered list of safety steps."""
    entry = ADVISORY_TABLE.get(alert_type, ADVISORY_TABLE["DEFAULT"])
    return {
        "alert_type": alert_type,
        "condition": entry["condition"],
        "severity": entry["severity"],
        "risk_level": risk_level,
        "steps": entry["steps"],
    }


def format_email_body(advisory: Dict[str, Any], telemetry: Dict[str, Any], gps: Dict[str, Any],
                      reason: str = None, trip_details: Dict[str, Any] = None) -> str:
    import datetime
    import html
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    trip = trip_details or {}
    driver_name = html.escape(str(trip.get("driver_name", "Registered Driver")))
    driver_email = html.escape(str(trip.get("driver_email", "Not Provided")))
    passenger_name = html.escape(str(trip.get("passenger_name", "Registered Passenger")))
    passenger_emergency_email = html.escape(str(trip.get("passenger_emergency_email", "emergency@contact.com")))
    vehicle_id = html.escape(str(trip.get("vehicle_id", "AP-04-REG-01")))
    trip_id = html.escape(str(trip.get("trip_id", telemetry.get("session_id", f"TRIP-{int(datetime.datetime.now().timestamp())}"))))
    incident_ref = f"DMS-EMERGENCY-{trip_id[-8:]}"

    lat = float(gps.get("lat", 0.0))
    lon = float(gps.get("lon", 0.0))
    maps_link = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
    nav_link = f"https://www.google.com/maps/dir/?api=1&destination={lat:.6f},{lon:.6f}&travelmode=driving"

    speed = float(telemetry.get("speed_kmh", 0.0))
    speed_text = f"{speed:.1f} km/h (In Motion)" if speed >= 1.0 else "0.0 km/h (Stationary / Stopped)"

    alert_reason = html.escape(str(reason or advisory.get("condition", "Automated Driver Safety Alert")))
    severity = html.escape(str(advisory.get("severity", "HIGH")))
    severity_color = "#dc2626" if severity == "CRITICAL" else ("#ea580c" if severity == "HIGH" else "#ca8a04")

    emotion = html.escape(str(telemetry.get("dominant_emotion", "Normal")))
    stress_pct = int(telemetry.get("stress_score", 0.0) * 100)
    risk_score = float(telemetry.get("risk_score", 0.0))
    risk_level = html.escape(str(advisory.get("risk_level", "Medium")))

    return f"""
    <div style="font-family: Arial, Helvetica, sans-serif; max-width: 680px; margin: 0 auto; background: #ffffff; color: #1e293b; border: 2px solid #0f172a; border-radius: 4px; padding: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.06);">
        <!-- Formal Dispatch Header -->
        <div style="border-bottom: 2px solid #0f172a; padding-bottom: 12px; margin-bottom: 18px;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td>
                        <div style="font-size: 19px; font-weight: bold; color: #0f172a; letter-spacing: 0.5px;">GUARDIANDRIVE AI TELEMATICS DISPATCH</div>
                        <div style="font-size: 13px; color: #475569; margin-top: 2px;">Automated In-Cabin Vehicle Safety & Emergency Dispatch System</div>
                    </td>
                    <td style="text-align: right; vertical-align: top;">
                        <div style="font-size: 12px; font-weight: bold; color: #dc2626; border: 1px solid #dc2626; padding: 4px 8px; display: inline-block;">
                            URGENT SAFETY NOTICE
                        </div>
                    </td>
                </tr>
            </table>
        </div>

        <!-- Recipient Notice -->
        <p style="font-size: 13px; margin: 0 0 14px 0; line-height: 1.6; color: #334155;">
            <b>TO:</b> Designated Passenger Emergency Contact (<b>{passenger_emergency_email}</b>)<br>
            <b>DATE & TIME:</b> {now_str}<br>
            <b>INCIDENT REFERENCE:</b> <code>{incident_ref}</code>
        </p>

        <div style="background: #fef2f2; border: 1px solid #fecaca; border-left: 4px solid #dc2626; padding: 14px 18px; margin-bottom: 20px; font-size: 14px;">
            <b>OFFICIAL NOTICE:</b> This is an automated emergency safety dispatch regarding passenger <b>{passenger_name}</b> and driver <b>{driver_name}</b> aboard vehicle <b>{vehicle_id}</b>. The vehicle's active monitoring system has detected an emergency safety event:
            <div style="margin-top: 6px; font-size: 16px; font-weight: bold; color: #991b1b;">
                ⚠️ {alert_reason}
            </div>
        </div>

        <!-- Section 1: Trip & Personnel Particulars Table -->
        <h3 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #0f172a; margin: 18px 0 8px 0; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px;">
            1. Trip &amp; Occupant Particulars
        </h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 18px; border: 1px solid #cbd5e1;">
            <tr style="background: #f8fafc;">
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; width: 30%;"><b>Passenger Name:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: bold;">{passenger_name}</td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; width: 25%;"><b>Emergency Email:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;">{passenger_emergency_email}</td>
            </tr>
            <tr>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><b>Driver Name:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: bold;">{driver_name}</td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><b>Driver Email / Contact:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;">{driver_email}</td>
            </tr>
            <tr style="background: #f8fafc;">
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><b>Vehicle Identifier:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: bold;">{vehicle_id}</td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><b>Trip Reference ID:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><code>{trip_id}</code></td>
            </tr>
        </table>

        <!-- Section 2: Incident Telemetry Table -->
        <h3 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #0f172a; margin: 18px 0 8px 0; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px;">
            2. Incident Telemetry &amp; Vehicle Status
        </h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 18px; border: 1px solid #cbd5e1;">
            <tr style="background: #f8fafc;">
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; width: 30%;"><b>Incident Reason:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: bold; color: #b91c1c;">{alert_reason}</td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; width: 25%;"><b>Severity Level:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: bold; color: {severity_color};">{severity}</td>
            </tr>
            <tr>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><b>Recorded Vehicle Speed:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1; font-weight: bold;">{speed_text}</td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><b>Safety Risk Index:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;">{risk_score:.2f} / 1.00 ({risk_level})</td>
            </tr>
            <tr style="background: #f8fafc;">
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><b>Driver Mood &amp; Stress:</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;">{stress_pct}% Stress ({emotion})</td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;"><b>Eye Aspect Ratio (EAR):</b></td>
                <td style="padding: 8px 12px; border: 1px solid #cbd5e1;">{telemetry.get('ear', 0.0):.2f}</td>
            </tr>
        </table>

        <!-- Section 3: Live GPS Coordinates & Google Maps Navigation -->
        <h3 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #0f172a; margin: 18px 0 8px 0; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px;">
            3. Live Vehicle Location &amp; Google Maps Navigation
        </h3>
        <div style="background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; padding: 16px; margin-bottom: 18px;">
            <div style="font-size: 13px; color: #334155;">
                <b>Current Geographic Coordinates:</b> Latitude: <b>{lat:.6f}</b>, Longitude: <b>{lon:.6f}</b>
            </div>
            <div style="margin-top: 14px;">
                <a href="{nav_link}" style="display: inline-block; background: #0f172a; color: #ffffff; padding: 11px 20px; text-decoration: none; font-weight: bold; font-size: 13px; border-radius: 4px; margin-right: 10px;">
                    🚗 Open Google Maps Turn-by-Turn Directions to Vehicle &rarr;
                </a>
                <a href="{maps_link}" style="display: inline-block; background: #ffffff; color: #0f172a; border: 1px solid #0f172a; padding: 10px 16px; text-decoration: none; font-weight: 600; font-size: 13px; border-radius: 4px;">
                    📍 View Exact Location Pin
                </a>
            </div>
        </div>

        <!-- Section 4: Recommended Action for Emergency Contact -->
        <h3 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #0f172a; margin: 18px 0 8px 0; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px;">
            4. Required Actions for Designated Emergency Contact
        </h3>
        <ol style="font-size: 13px; line-height: 1.6; color: #334155; margin: 0 0 20px 0; padding-left: 20px;">
            <li><b>Attempt Immediate Telephone Contact:</b> Reach out to driver <b>{driver_name}</b> or passenger <b>{passenger_name}</b> immediately to confirm consciousness and road safety.</li>
            <li><b>Verify Safe Vehicle Halting:</b> If driver is experiencing drowsiness or distracted phone usage, advise them to bring the vehicle to an immediate safe stop.</li>
            <li><b>Track Live Vehicle Location:</b> Use the Google Maps navigation button above to track vehicle location or direct assistance.</li>
            <li><b>Escalate to Emergency Services:</b> If the occupants are unresponsive and the vehicle is in high-speed motion, notify local highway emergency services quoting Vehicle ID <b>{vehicle_id}</b> and Coordinates <b>{lat:.6f}, {lon:.6f}</b>.</li>
        </ol>

        <!-- Formal Sign-Off / Footer -->
        <div style="border-top: 2px solid #0f172a; padding-top: 12px; margin-top: 20px; font-size: 11px; color: #64748b; line-height: 1.5;">
            <b>GuardianDrive AI Transportation Telematics Division</b><br>
            Official Incident Dispatch Notice &bull; Generated automatically from onboard vehicle sensors.<br>
            Confidential: Intended solely for designated passenger emergency contacts.
        </div>
    </div>
    """


def format_email_text(advisory: Dict[str, Any], telemetry: Dict[str, Any], gps: Dict[str, Any],
                      reason: str = None, trip_details: Dict[str, Any] = None) -> str:
    """Formats the plain-text fallback version for traditional terminals and mobile lock screen previews."""
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    trip = trip_details or {}
    driver_name = trip.get("driver_name", "Registered Driver")
    driver_email = trip.get("driver_email", "Not Provided")
    passenger_name = trip.get("passenger_name", "Registered Passenger")
    passenger_emergency_email = trip.get("passenger_emergency_email", "emergency@contact.com")
    vehicle_id = trip.get("vehicle_id", "AP-04-REG-01")
    trip_id = trip.get("trip_id", telemetry.get("session_id", f"TRIP-{int(datetime.datetime.now().timestamp())}"))
    incident_ref = f"DMS-EMERGENCY-{trip_id[-8:]}"

    lat = float(gps.get("lat", 0.0))
    lon = float(gps.get("lon", 0.0))
    maps_link = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
    nav_link = f"https://www.google.com/maps/dir/?api=1&destination={lat:.6f},{lon:.6f}&travelmode=driving"

    speed = float(telemetry.get("speed_kmh", 0.0))
    speed_text = f"{speed:.1f} km/h (In Motion)" if speed >= 1.0 else "0.0 km/h (Stationary / Stopped)"
    alert_reason = reason or advisory.get("condition", "Automated Driver Safety Alert")
    severity = advisory.get("severity", "HIGH")
    risk_score = float(telemetry.get("risk_score", 0.0))
    risk_level = advisory.get("risk_level", "Medium")

    return f"""================================================================================
GUARDIANDRIVE AI TELEMATICS DISPATCH — URGENT SAFETY NOTICE
Automated In-Cabin Vehicle Safety & Emergency Dispatch System
================================================================================

TO: Designated Passenger Emergency Contact ({passenger_emergency_email})
DATE & TIME: {now_str}
INCIDENT REFERENCE: {incident_ref}

OFFICIAL NOTICE:
This is an automated emergency safety dispatch regarding passenger {passenger_name}
and driver {driver_name} aboard vehicle {vehicle_id}. The vehicle's active
monitoring system has detected an emergency safety event:
>>> {alert_reason} <<<

--------------------------------------------------------------------------------
1. TRIP & OCCUPANT PARTICULARS
--------------------------------------------------------------------------------
Passenger Name:              {passenger_name}
Passenger Emergency Email:   {passenger_emergency_email}
Driver Name:                 {driver_name}
Driver Email / Phone:        {driver_email}
Vehicle Identifier:          {vehicle_id}
Trip Reference ID:           {trip_id}

--------------------------------------------------------------------------------
2. INCIDENT TELEMETRY & VEHICLE STATUS
--------------------------------------------------------------------------------
Incident Reason:             {alert_reason}
Severity Level:              {severity}
Recorded Vehicle Speed:      {speed_text}
Safety Risk Index:           {risk_score:.2f} / 1.00 ({risk_level})
Eye Aspect Ratio (EAR):      {telemetry.get('ear', 0.0):.2f}

--------------------------------------------------------------------------------
3. LIVE VEHICLE LOCATION & GOOGLE MAPS NAVIGATION
--------------------------------------------------------------------------------
Geographic Coordinates:      Latitude {lat:.6f}, Longitude {lon:.6f}
Google Maps Turn-by-Turn:    {nav_link}
Location Pin:                {maps_link}

--------------------------------------------------------------------------------
4. REQUIRED ACTIONS FOR DESIGNATED EMERGENCY CONTACT
--------------------------------------------------------------------------------
1. Attempt Immediate Telephone Contact: Reach out to driver {driver_name} or
   passenger {passenger_name} immediately to confirm consciousness and road safety.
2. Verify Safe Vehicle Halting: If driver is experiencing drowsiness or phone
   distraction, advise them to bring the vehicle to an immediate safe stop.
3. Track Live Vehicle Location: Use the Google Maps link above to track vehicle.
4. Escalate to Emergency Services: If occupants are unresponsive and the vehicle
   is in motion, notify emergency services with Vehicle ID {vehicle_id} and
   coordinates {lat:.6f}, {lon:.6f}.

================================================================================
GuardianDrive AI Transportation Telematics Division
Official Incident Dispatch Notice • Confidential
================================================================================
"""

