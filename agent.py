"""
agent.py — Background keystroke dynamics agent.
Captures ONLY key timing. No key identities, no text content.
Lifecycle: training → active
"""

import time
import threading
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable

import model
import logger
import mailer
from features import extract_features, FEATURE_DIM

DATA_DIR        = Path("data")
TRAIN_DATA_PATH = DATA_DIR / "training_data.npy"

WINDOW_SECONDS    = 10
MIN_EVENTS        = 15
MIN_SAMPLES_TRAIN = 8
RETRAIN_EVERY     = 15
VARIANCE_THRESHOLD = 0.3

_score_history: List[float] = []
MAX_SCORE_HISTORY = 60


class AgentState:
    def __init__(self):
        self.mode           = "training"
        self.running        = False
        self._events        = []
        self._window_start  = time.time()
        self._lock          = threading.Lock()
        self._training_data = np.empty((0, FEATURE_DIM), dtype=np.float32)
        self._new_windows   = 0
        self.sample_count   = 0
        self.window_count   = 0
        self.retrain_count  = 0
        self.last_confidence = -1.0
        self.last_score_time = 0.0
        self.alerts_sent    = 0
        self.streak_anomaly = 0
        self.smoothed_conf  = -1.0
        self._alpha         = 0.25


_state = AgentState()
_listener = None
_cfg: dict = {}


def start():
    global _listener
    if _state.running:
        return
    DATA_DIR.mkdir(exist_ok=True)
    _load_training_data()
    model.load()
    _state.running = True
    from pynput import keyboard as kb
    _listener = kb.Listener(on_press=_on_press, on_release=_on_release)
    _listener.start()
    logger.log("START", {"mode": _state.mode})
    threading.Thread(target=_window_loop, daemon=True).start()


def stop():
    global _listener
    _state.running = False
    if _listener:
        _listener.stop()
        _listener = None
    logger.log("STOP", {})


def set_mode(mode: str):
    if mode not in ("training", "active"):
        raise ValueError(f"Unknown mode: {mode}")
    prev = _state.mode
    _state.mode = mode
    logger.log("MODE_CHANGE", {"from": prev, "to": mode})


def set_config(cfg: dict):
    global _cfg
    _cfg = cfg


def get_live() -> dict:
    history = list(_score_history[-30:])
    avg     = float(np.mean(history)) if history else -1.0
    trend   = float(history[-1] - history[-10]) if len(history) >= 10 else 0.0
    return {
        "confidence":     round(_state.last_confidence, 4),
        "smoothed":       round(_state.smoothed_conf, 4),
        "avg_confidence": round(avg, 4),
        "trend":          round(trend, 4),
        "mode":           _state.mode,
        "samples":        _state.sample_count,
        "windows":        _state.window_count,
        "retrain_count":  _state.retrain_count,
        "alerts_sent":    _state.alerts_sent,
        "streak_anomaly": _state.streak_anomaly,
        "model_trained":  model.is_trained(),
        "score_history":  [round(s, 3) for s in history],
        "timestamp":      time.time(),
    }


def get_readiness() -> dict:
    trained = model.is_trained()
    conf    = _state.smoothed_conf if trained else -1.0
    score   = 0
    checks  = []

    if _state.sample_count >= 8:
        score += 30
        checks.append({"name": "Sufficient samples", "ok": True,  "detail": f"{_state.sample_count} collected"})
    else:
        checks.append({"name": "Sufficient samples", "ok": False, "detail": f"{_state.sample_count}/8 needed"})

    if trained:
        score += 30
        checks.append({"name": "Model trained",      "ok": True,  "detail": f"{_state.retrain_count} retrains"})
    else:
        checks.append({"name": "Model trained",      "ok": False, "detail": "Not trained yet"})

    if conf >= 0.65:
        score += 25
        checks.append({"name": "Confidence healthy", "ok": True,  "detail": f"{conf*100:.0f}% ≥ 65%"})
    elif conf >= 0:
        checks.append({"name": "Confidence healthy", "ok": False, "detail": f"{conf*100:.0f}% < 65%"})
    else:
        checks.append({"name": "Confidence healthy", "ok": False, "detail": "No score yet"})

    has_email = bool(_cfg.get("email_sender") and _cfg.get("email_password") and _cfg.get("email_recipient"))
    if has_email:
        score += 15
        checks.append({"name": "Email configured",   "ok": True,  "detail": _cfg.get("email_recipient", "")})
    else:
        checks.append({"name": "Email configured",   "ok": False, "detail": "Not configured"})

    return {"score": score, "ready": score >= 70, "checks": checks}


def _on_press(key):
    with _state._lock:
        _state._events.append(("press", time.time() * 1000))


def _on_release(key):
    with _state._lock:
        _state._events.append(("release", time.time() * 1000))


def _window_loop():
    while _state.running:
        time.sleep(WINDOW_SECONDS)
        _flush_window()


def _flush_window():
    now = time.time()
    with _state._lock:
        events = list(_state._events)
        _state._events.clear()
        _state._window_start = now

    if len(events) < MIN_EVENTS:
        return

    vec = extract_features(events)
    if vec is None:
        return

    _state.window_count += 1

    if _state.mode == "training":
        _accumulate(vec)
        # FIX 1: score in training mode too once model exists
        if model.is_trained():
            _update_confidence(model.score(vec))

    elif _state.mode == "active":
        if model.is_trained():
            raw = model.score(vec)
            _update_confidence(raw)
            logger.log("SCORE", {"confidence": round(raw, 4), "smoothed": round(_state.smoothed_conf, 4)})
            _check_anomaly(_state.smoothed_conf)
        _accumulate(vec)


def _update_confidence(raw: float):
    _state.last_confidence = raw
    _state.last_score_time = time.time()
    if _state.smoothed_conf < 0:
        _state.smoothed_conf = raw
    else:
        _state.smoothed_conf = _state._alpha * raw + (1 - _state._alpha) * _state.smoothed_conf
    _score_history.append(_state.smoothed_conf)
    if len(_score_history) > MAX_SCORE_HISTORY:
        _score_history.pop(0)


def _check_anomaly(conf: float):
    threshold = _cfg.get("alert_threshold", 0.35)
    if conf < threshold:
        _state.streak_anomaly += 1
        logger.log("ANOMALY", {"confidence": round(conf, 4), "streak": _state.streak_anomaly})
        _maybe_send_alert(conf, "high")
    elif conf < 0.65 and _cfg.get("alert_warn", False):
        _state.streak_anomaly = 0
        _maybe_send_alert(conf, "warn")
    else:
        _state.streak_anomaly = 0


def _maybe_send_alert(conf: float, level: str):
    if not _cfg:
        return
    username = _cfg.get("username", "User")
    pct      = f"{conf*100:.1f}%"

    if level == "high" and _cfg.get("alert_high", True):
        subject = f"High Alert — Identity Failure ({pct})"
        body    = (f"Hello {username},\n\nKeyGuard detected a LOW CONFIDENCE match.\n\n"
                   f"Confidence: {pct} (threshold: {_cfg.get('alert_threshold',0.35)*100:.0f}%)\n"
                   f"Streak: {_state.streak_anomaly} consecutive windows\n\n— KeyGuard")
        mailer.send_alert_async(_cfg, subject, body, alert_type="high_alert")
        _state.alerts_sent += 1

    elif level == "warn" and _cfg.get("alert_warn", False):
        subject = f"Warning — Marginal Identity Match ({pct})"
        body    = (f"Hello {username},\n\nKeyGuard detected a MARGINAL score.\n\n"
                   f"Confidence: {pct} (trusted threshold: 65%)\n\n— KeyGuard")
        mailer.send_alert_async(_cfg, subject, body, alert_type="warn_alert")
        _state.alerts_sent += 1


def _accumulate(vec: np.ndarray):
    _state._training_data = np.vstack([_state._training_data, vec.reshape(1, -1)])
    _state.sample_count  += 1
    _state._new_windows  += 1
    _save_training_data()

    if _state.sample_count >= MIN_SAMPLES_TRAIN and not model.is_trained():
        _retrain()
        return
    if _state._new_windows >= RETRAIN_EVERY:
        _retrain()


def _retrain():
    X = _state._training_data
    if float(np.mean(np.var(X, axis=0))) < VARIANCE_THRESHOLD:
        logger.log("RETRAIN_SKIP", {"reason": "low_variance"})
        return
    if model.train(X):
        _state.retrain_count += 1
        _state._new_windows   = 0
        logger.log("RETRAIN", {"samples": len(X), "retrain_count": _state.retrain_count})
        if _cfg.get("alert_retrain") and _cfg.get("email_recipient"):
            body = (f"Hello {_cfg.get('username','User')},\n\nModel updated.\n\n"
                    f"Samples: {len(X)}\nRetrains: {_state.retrain_count}\n\n— KeyGuard")
            mailer.send_alert_async(_cfg, "Model Retrained", body, alert_type="retrain")


def _save_training_data():
    np.save(str(TRAIN_DATA_PATH), _state._training_data)


def _load_training_data():
    if TRAIN_DATA_PATH.exists():
        try:
            data = np.load(str(TRAIN_DATA_PATH))
            if data.ndim == 2 and data.shape[1] == FEATURE_DIM:
                _state._training_data = data.astype(np.float32)
                _state.sample_count   = len(data)
        except Exception:
            pass
    _state._training_data = _state._training_data if _state.sample_count else np.empty((0, FEATURE_DIM), dtype=np.float32)