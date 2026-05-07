import json, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH   = Path("data/logs.json")
MAX_EVENTS = 500

_lock   = threading.Lock()
_buffer: list[dict] = []

VALID_EVENTS = {
    "START", "STOP", "SCORE", "RETRAIN", "RETRAIN_SKIP",
    "ANOMALY", "ALERT", "ALERT_SENT", "MODE_CHANGE",
    "CONFIG_SAVE", "KILL_DETECTED", "RESET", "ERROR",
}

def log(event: str, meta: dict[str, Any] | None = None):
    entry = {"time": datetime.now(timezone.utc).isoformat(), "event": event}
    if meta:
        entry.update(meta)
    with _lock:
        _buffer.append(entry)
        if len(_buffer) > MAX_EVENTS:
            _buffer.pop(0)
    threading.Thread(target=_persist, daemon=True).start()

def recent(n: int = 50) -> list[dict]:
    with _lock:
        return list(_buffer[-n:])

def load():
    global _buffer
    if LOG_PATH.exists():
        try:
            data = json.loads(LOG_PATH.read_text())
            if isinstance(data, list):
                _buffer = data[-MAX_EVENTS:]
                return
        except Exception:
            pass
    _buffer = []

def _persist():
    LOG_PATH.parent.mkdir(exist_ok=True)
    try:
        with _lock:
            snapshot = list(_buffer)
        with open(LOG_PATH, "w") as f:
            json.dump(snapshot, f)
    except Exception:
        pass

load()