"""
mailer.py — Email alert delivery via Gmail SMTP + App Password.

Uses Python's built-in smtplib — no third-party email library needed.
The App Password is read from config each call so hot-reloads work.
"""

import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from datetime             import datetime
from pathlib              import Path

import logger

_send_lock  = threading.Lock()
_last_sent: dict[str, float] = {}   # alert_type → epoch seconds
DEBOUNCE_S  = 60                    # min seconds between same alert type


def send_alert(
    cfg: dict,
    subject: str,
    body_text: str,
    alert_type: str = "generic",
    force: bool = False,
) -> bool:
    """
    Send an email alert.

    Args:
        cfg:        config dict with email_sender / email_password / email_recipient
        subject:    email subject line
        body_text:  plain-text body
        alert_type: dedup key (e.g. 'high_alert', 'warning', 'retrain')
        force:      bypass debounce (used for test emails)

    Returns True on success, False on failure.
    """
    sender    = cfg.get("email_sender",    "").strip()
    password  = cfg.get("email_password",  "").strip()
    recipient = cfg.get("email_recipient", "").strip()
    username  = cfg.get("username",        "User")

    if not sender or not password or not recipient:
        logger.log("ALERT_SENT", {"status": "skipped", "reason": "no_credentials"})
        return False

    # Debounce
    import time
    now = time.time()
    if not force and (now - _last_sent.get(alert_type, 0)) < DEBOUNCE_S:
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[KeyGuard] {subject}"
    msg["From"]    = f"KeyGuard <{sender}>"
    msg["To"]      = recipient

    html_body = _make_html(username, subject, body_text)

    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with _send_lock:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
                smtp.login(sender, password.replace(" ", ""))
                smtp.sendmail(sender, recipient, msg.as_string())

        _last_sent[alert_type] = now
        logger.log("ALERT_SENT", {
            "to":      recipient,
            "subject": subject,
            "type":    alert_type,
        })
        return True

    except smtplib.SMTPAuthenticationError:
        logger.log("ALERT_SENT", {"status": "auth_error", "hint": "Check App Password"})
        return False
    except Exception as e:
        logger.log("ALERT_SENT", {"status": "error", "detail": str(e)})
        return False


def send_alert_async(cfg: dict, subject: str, body: str, alert_type: str = "generic", force: bool = False):
    """Fire-and-forget email (background thread)."""
    t = threading.Thread(
        target=send_alert,
        args=(cfg, subject, body, alert_type, force),
        daemon=True,
    )
    t.start()


# ─── HTML template ────────────────────────────────────────────────────────

def _make_html(username: str, subject: str, body: str) -> str:
    body_html = body.replace("\n", "<br>")
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#06060a;margin:0;padding:32px}}
  .wrap{{max-width:520px;margin:0 auto}}
  .card{{background:#0e0e18;border:1px solid rgba(124,109,250,0.25);border-radius:14px;overflow:hidden}}
  .hdr{{background:linear-gradient(135deg,#7c6dfa,#5f52e0);padding:24px 28px}}
  .hdr-title{{color:#fff;font-size:20px;font-weight:700;letter-spacing:-0.5px;margin:0}}
  .hdr-sub{{color:rgba(255,255,255,0.7);font-size:12px;margin-top:4px}}
  .body{{padding:28px}}
  .body p{{color:#c8c8d8;font-size:14px;line-height:1.7;margin:0 0 12px}}
  .foot{{padding:16px 28px;border-top:1px solid rgba(255,255,255,0.06);
         font-size:11px;color:#44445a;display:flex;justify-content:space-between}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:20px;
          font-size:11px;font-weight:700;letter-spacing:0.5px}}
  .badge-danger{{background:rgba(244,63,94,0.15);color:#f43f5e;border:1px solid rgba(244,63,94,0.3)}}
  .badge-warn{{background:rgba(245,158,11,0.12);color:#f59e0b;border:1px solid rgba(245,158,11,0.25)}}
  .badge-ok{{background:rgba(16,185,129,0.1);color:#10b981;border:1px solid rgba(16,185,129,0.25)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="hdr">
      <div class="hdr-title">🔐 KeyGuard Alert</div>
      <div class="hdr-sub">{subject}</div>
    </div>
    <div class="body">
      <p>Hello <strong style="color:#e8e8f0">{username}</strong>,</p>
      <p>{body_html}</p>
    </div>
    <div class="foot">
      <span>KeyGuard Behavioral Auth</span>
      <span>{ts}</span>
    </div>
  </div>
</div>
</body>
</html>"""