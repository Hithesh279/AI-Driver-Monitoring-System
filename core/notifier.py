"""
Email Alert Notifier.

Sends the driver's condition, recommended safety steps, and live GPS
location to configured recipients (e.g. family / fleet manager) using
plain SMTP (smtplib) — no third-party service or account beyond a normal
email account (e.g. a Gmail app password) is required.

All settings come from environment variables (see config.py, section 8) —
never hardcode credentials in source.
"""

import smtplib
import time
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from config import (
    EMAIL_ENABLED, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    ALERT_RECIPIENT_EMAILS, EMAIL_COOLDOWN_SEC
)


class EmailNotifier:
    def __init__(self):
        self.enabled = EMAIL_ENABLED and bool(SMTP_USER) and bool(SMTP_PASSWORD)
        self.smtp_user = SMTP_USER
        self.default_recipients = [
            e.strip() for e in ALERT_RECIPIENT_EMAILS 
            if e.strip() and "example.com" not in e and "your_" not in e
        ]
        self._last_sent_by_key: Dict[str, float] = {}
        self._lock = threading.Lock()
        if EMAIL_ENABLED and not self.enabled:
            print("[EmailNotifier] DMS_EMAIL_ENABLED=true but SMTP_USER/PASSWORD "
                  "are not fully configured — email alerts disabled.")

    def send_alert(self, subject: str, html_body: str, key: str = None, cooldown: float = None,
                   recipients: Optional[List[str]] = None, text_body: Optional[str] = None) -> bool:
        """Sends a multi-part alternative email (both plain text & rich HTML) in a background thread."""
        if not self.enabled:
            return False

        # Gather clean recipient emails
        raw_list = recipients if (recipients is not None and len(recipients) > 0) else self.default_recipients
        target_recipients = [
            r.strip() for r in raw_list 
            if r and r.strip() and "example.com" not in r and "your_" not in r
        ]
        # If no recipient specified, fallback to sending to the authenticated SMTP user itself
        if not target_recipients and self.smtp_user and "@" in self.smtp_user:
            target_recipients = [self.smtp_user]

        if not target_recipients:
            return False

        key = key or subject
        gap = EMAIL_COOLDOWN_SEC if cooldown is None else cooldown
        now = time.time()
        with self._lock:
            last = self._last_sent_by_key.get(key, 0.0)
            if now - last < gap:
                return False
            self._last_sent_by_key[key] = now
        threading.Thread(target=self._send, args=(subject, html_body, text_body, target_recipients), daemon=True).start()
        return True

    def _send(self, subject: str, html_body: str, text_body: Optional[str], recipients: List[str]):
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = SMTP_USER
            msg["To"] = ", ".join(recipients)

            # 1. Plain Text fallback part (RFC 2046: lower fidelity first)
            plain_fallback = text_body or "GUARDIANDRIVE AI EMERGENCY DISPATCH: Please view in an HTML-capable email client."
            msg.attach(MIMEText(plain_fallback, "plain", "utf-8"))

            # 2. Rich HTML part (primary rendering for modern mail clients)
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, recipients, msg.as_string())
            print(f"[EmailNotifier] Emergency notification successfully sent to: {', '.join(recipients)}")
        except Exception as e:
            print(f"[EmailNotifier] Failed to send alert email: {e}")
