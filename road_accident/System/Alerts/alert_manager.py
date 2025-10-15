# System/Alerts/alert_manager.py
import time
import threading
from typing import Dict, Tuple, List, Any, Optional

from System.Alerts.fire_rescue_api import send_fire_rescue_alert
from System.Alerts.notifier import send_sms, send_email

# If you want to escalate when nobody acknowledges in X seconds
ALERT_TIMEOUT_SEC = 30

def trigger_alert(
    event_name: str,
    city: str,
    district: int,
    camera_id: str,
    frame_id: int,
    crash_dims: List[List[int]],
    gps: Optional[Tuple[float, float]] = None,
    snapshot_b64: Optional[str] = None,   # you can pass a base64 thumbnail if you already generate one
) -> None:
    """
    Fire the alert fan-out when a crash is detected.
    Safe to call from Master.checkResult() right after it decides it's a crash.
    """
    alert = {
        "event": event_name,
        "city": city,
        "district": district,
        "camera_id": camera_id,
        "frame_id": frame_id,
        "crash_boxes": crash_dims,   # [[x, y, w, h], ...] or [ymin,xmin,ymax,xmax] per your format
        "gps": gps,
        "snapshot_b64": snapshot_b64,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ack": False,
    }

    # 1) Notify Fire & Rescue “control room”
    send_fire_rescue_alert(alert)

    # 2) Optional: local SMS/Email fan-out
    try:
        loc = f"{city}, District {district}"
        send_sms(f"Accident detected @ {loc} | Camera {camera_id} | Frame {frame_id}")
        send_email(
            subject=f"URGENT: Accident detected @ {loc}",
            body=(
                f"Event: {event_name}\n"
                f"City: {city}\nDistrict: {district}\n"
                f"Camera: {camera_id}\nFrame: {frame_id}\n"
                f"Time: {alert['ts']}\n"
                f"Boxes: {crash_dims}\n"
                f"GPS: {gps if gps else 'N/A'}\n"
            )
        )
    except Exception as e:
        print(f"[Alerts] Local notifier failed: {e}")

    # 3) Background escalation if no acknowledgment
    threading.Thread(target=_await_ack_and_escalate, args=(alert,), daemon=True).start()


def _await_ack_and_escalate(alert: Dict[str, Any]) -> None:
    """
    Simple timer-based escalation.
    In a real system, 'ack' would be toggled by a REST endpoint/UI when responders acknowledge.
    """
    time.sleep(ALERT_TIMEOUT_SEC)
    if not alert.get("ack"):
        print("[Alerts] No acknowledgment received; escalating...")
        try:
            send_email(
                subject="ESCALATION: Unacknowledged Accident",
                body=(
                    f"Event: {alert['event']}\n"
                    f"City: {alert['city']}\nDistrict: {alert['district']}\n"
                    f"Camera: {alert['camera_id']}\nFrame: {alert['frame_id']}\n"
                    f"Time: {alert['ts']}\n"
                    f"Crash Boxes: {alert.get('crash_boxes')}\n"
                ),
                to="",
            )
        except Exception as e:

            print(f"[Alerts] Escalation email failed: {e}")
