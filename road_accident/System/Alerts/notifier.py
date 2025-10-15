# System/Alerts/notifier.py
import os
import json
import smtplib
import requests
import certifi
import urllib3
from email.mime.text import MIMEText
from datetime import datetime
from typing import Optional, Any, Dict

# ==========================================================
# 🔧 CONFIGURATION — EDIT THESE
# ==========================================================

# --- Gmail email setup ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT   = 465
SMTP_USER   = "sathyasandars@gmail.com"         # ← your Gmail
SMTP_PASS   = "yvft gity hqch uzay"      # ← Gmail App Password
ALERT_FROM  = SMTP_USER
DEFAULT_EMAIL_TO = "sathyasandar.s.2024.mtechds@rajalakshmi.edu.in"

# --- Vonage SMS setup ---
VONAGE_API_KEY    = "fd4a4e36"    # ← from dashboard.vonage.com
VONAGE_API_SECRET = "RWbET1GkYwVD1eP3"
VONAGE_FROM       = "AccidentAlert"             # up to 11 chars (sender id)
DEFAULT_SMS_TO    = "918682064563"           # no '+'; e.g., 919876543210

# Network/Proxy/TLS toggles
USE_PROXY_FROM_ENV = True                    # honors HTTPS_PROXY/HTTP_PROXY if set
ALLOW_INSECURE_FALLBACK = True               # last-resort verify=False if CAs fail

# Vonage REST endpoint for classic SMS API
VONAGE_SMS_URL = "https://rest.nexmo.com/sms/json"

# ==========================================================
# 📧 EMAIL ALERTS
# ==========================================================

def send_email(subject: str, body: str, to: str = DEFAULT_EMAIL_TO) -> None:
    """Send alert email via Gmail SMTP."""
    if not (SMTP_USER and SMTP_PASS):
        print(f"[EMAIL SIMULATION] {subject}\n{body}")
        return

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = ALERT_FROM
    msg["To"] = to

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=30) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(ALERT_FROM, [to], msg.as_string())
        print(f"📧 Email sent successfully → {to}")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")


# ==========================================================
# 📱 VONAGE SMS (robust TLS handling)
# ==========================================================

def _make_session(verify: Optional[Any]) -> requests.Session:
    """
    Create a requests Session honoring proxies (if enabled) and the given verify setting.
    verify can be: True/False or a path to a CA bundle (e.g., certifi.where()).
    """
    sess = requests.Session()
    sess.verify = verify
    if USE_PROXY_FROM_ENV:
        # requests picks these up automatically; we copy explicitly to be clear.
        proxies = {}
        if os.getenv("HTTPS_PROXY"):
            proxies["https"] = os.getenv("HTTPS_PROXY")
        if os.getenv("HTTP_PROXY"):
            proxies["http"] = os.getenv("HTTP_PROXY")
        if proxies:
            sess.proxies.update(proxies)
    return sess


def _post_vonage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try Vonage POST with three strategies:
      1) Default trust store
      2) certifi bundle
      3) (optional) verify=False
    Returns parsed JSON response or raises the last exception.
    """
    last_err = None

    # 1) Default platform CAs
    try:
        sess = _make_session(verify=True)
        r = sess.post(VONAGE_SMS_URL, data=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        last_err = e
        print(f"⚠️ Vonage try#1 (system CA) failed: {e}")

    # 2) Force certifi CA bundle (often fixes Windows/corporate CA issues)
    try:
        sess = _make_session(verify=certifi.where())
        r = sess.post(VONAGE_SMS_URL, data=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        last_err = e
        print(f"⚠️ Vonage try#2 (certifi CA) failed: {e}")

    # 3) Last resort: disable verification (only if allowed)
    if ALLOW_INSECURE_FALLBACK:
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            sess = _make_session(verify=False)
            r = sess.post(VONAGE_SMS_URL, data=payload, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            print(f"⚠️ Vonage try#3 (verify=False) failed: {e}")

    # If all attempts failed, raise the last error
    raise last_err


def send_sms(message: str, to: str = DEFAULT_SMS_TO) -> None:
    """Send SMS using Vonage REST API with resilient TLS path."""
    if not (VONAGE_API_KEY and VONAGE_API_SECRET and to):
        print(f"[SMS SIMULATION] {message} → {to}")
        return

    payload = {
        "api_key": VONAGE_API_KEY,
        "api_secret": VONAGE_API_SECRET,
        "to": to,
        "from": VONAGE_FROM,
        "text": message,
    }

    try:
        resp = _post_vonage(payload)
        # Vonage response: {'messages':[{'status':'0','message-id':...,'to':...,'remaining-balance':...,'message-price':...,'network':...}]}
        msg = resp.get("messages", [{}])[0]
        status = msg.get("status")
        if status == "0":
            print(f"📱 SMS sent successfully → {to}")
        else:
            err = msg.get("error-text", "Unknown error")
            print(f"❌ Vonage error: {err} (status={status})")
    except Exception as e:
        print(f"❌ SMS sending failed (all attempts): {e}")


# ==========================================================
# 🚨 Combined Crash Alert
# ==========================================================

def trigger_alert(event_name: str,
                  city: str,
                  district: int,
                  camera_id: str,
                  frame_id: int,
                  crash_dims,
                  gps: Optional[Any] = None,
                  snapshot_b64: Optional[str] = None,
                  email_to: str = DEFAULT_EMAIL_TO,
                  phone_to: str = DEFAULT_SMS_TO) -> None:
    """Called by Master when a crash is detected."""
    payload = {
        "event": event_name or "Vehicle Collision",
        "city": city,
        "district": int(district),
        "camera_id": str(camera_id),
        "frame_id": int(frame_id),
        "crash_boxes": crash_dims,
        "gps": gps,
        "snapshot_b64": snapshot_b64,
        "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "ack": False,
    }

    print("🔥 [Fire&Rescue] Dispatching units with payload:")
    print(json.dumps(payload, indent=2))

    # Email
    subject = f"URGENT: Accident detected @ {city}"
    body = (
        f"Event: {payload['event']}\n"
        f"City: {city}\n"
        f"District: {district}\n"
        f"Camera ID: {camera_id}\n"
        f"Frame ID: {frame_id}\n"
        f"Time: {payload['ts']}\n"
        f"GPS: {gps}\n"
    )
    send_email(subject, body, to=email_to)

    # SMS
    sms_text = f"URGENT: Accident Detected🚨 : Crash @ {city} | Cam {camera_id} | Frame {frame_id} | Time {payload['ts']}"
    send_sms(sms_text, to=phone_to)


# ==========================================================
# 🧪 LOCAL TEST (run this file directly)
# ==========================================================
if __name__ == "__main__":
    send_email("🚨 Test Accident Alert", "This is a test crash alert from control room.")
    send_sms("🚨 Test SMS: crash detection system active.")