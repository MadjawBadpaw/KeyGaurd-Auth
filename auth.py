import bcrypt, json
from pathlib import Path

CONFIG_PATH = Path("data/config.json")

def _hash(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def is_set() -> bool:
    try:
        return bool(json.loads(CONFIG_PATH.read_text()).get("guardian_password_hash"))
    except Exception:
        return False

def set_password(pw: str):
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    cfg["guardian_password_hash"] = _hash(pw)
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

def verify(pw: str) -> bool:
    try:
        cfg    = json.loads(CONFIG_PATH.read_text())
        stored = cfg.get("guardian_password_hash", "")
        return bcrypt.checkpw(pw.encode(), stored.encode())
    except Exception:
        return False