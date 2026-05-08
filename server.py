import asyncio, json, sys
from pathlib import Path
from typing import Optional
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent, logger, mailer, auth

STATIC_DIR = Path("static")

# In-memory session: one token, one password per run
_session_token: Optional[str] = None
_session_pw:    Optional[str] = None   # kept in memory ONLY, never written

app = FastAPI(title="KeyGuard")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic models ──────────────────────────────────────────────────────

class ConfigPayload(BaseModel):
    username:        str   = "user"
    mode:            str   = "training"
    email_sender:    str   = ""
    email_password:  str   = ""
    email_recipient: str   = ""
    alert_threshold: float = 0.35
    alert_high:      bool  = True
    alert_warn:      bool  = False
    alert_retrain:   bool  = False
    contamination:   float = 0.05

class ModePayload(BaseModel):
    mode:     str
    password: Optional[str] = None
    token:    Optional[str] = None

class PasswordPayload(BaseModel):
    password: str

class SetupPayload(BaseModel):
    password: str
    confirm:  str

class TestEmailPayload(BaseModel):
    email_sender:    str
    email_password:  str
    email_recipient: str
    username:        str = "User"

VALID_MODES = {"training", "active"}


# ── helpers ──────────────────────────────────────────────────────────────

def _require_session(token: Optional[str]):
    if not token or token != _session_token:
        raise HTTPException(403, "Not authenticated")

def _pw_from_token() -> str:
    return _session_pw or ""


# ── auth routes ──────────────────────────────────────────────────────────

@app.get("/auth/status")
async def auth_status():
    return {"password_set": auth.is_set()}

@app.post("/auth/setup")
async def auth_setup(payload: SetupPayload):
    if auth.is_set():
        raise HTTPException(400, "Password already set")
    if payload.password != payload.confirm:
        raise HTTPException(400, "Passwords do not match")
    if len(payload.password) < 6:
        raise HTTPException(400, "Password too short (min 6 chars)")
    auth.set_password(payload.password)
    return {"status": "ok"}

@app.post("/auth/login")
async def auth_login(payload: PasswordPayload):
    global _session_token, _session_pw
    if not auth.verify(payload.password):
        raise HTTPException(403, "Invalid password")
    _session_token = secrets.token_hex(32)
    _session_pw    = payload.password   # held in RAM only
    return {"token": _session_token}

@app.post("/auth/verify")
async def auth_verify(payload: PasswordPayload):
    if not auth.verify(payload.password):
        raise HTTPException(403, "Invalid password")
    return {"status": "ok"}


# ── agent routes ─────────────────────────────────────────────────────────

@app.get("/")
async def index():
    p = STATIC_DIR / "index.html"
    return FileResponse(str(p)) if p.exists() else JSONResponse({"status": "place index.html in static/"})

@app.get("/agent/live")
async def live(): return agent.get_live()

@app.get("/agent/stream")
async def stream():
    async def gen():
        while True:
            try:
                yield f"data: {json.dumps(agent.get_live())}\n\n"
                await asyncio.sleep(1.0)
            except asyncio.CancelledError: break
            except Exception: await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/agent/logs")
async def logs(n: int = 100): return logger.recent(n)

@app.get("/agent/readiness")
async def readiness(): return agent.get_readiness()

@app.get("/config")
async def get_config(token: Optional[str] = None):
    _require_session(token)
    cfg = auth.load_config(_pw_from_token())
    cfg.pop("email_password", None)   # never send to frontend
    cfg["email_password_set"] = bool(auth.get_email_password())
    return cfg

@app.post("/config")
async def post_config(payload: ConfigPayload, token: Optional[str] = None):
    _require_session(token)
    cfg = payload.dict()
    auth.save_config(cfg, _pw_from_token())
    agent.set_config({**cfg, "email_password": auth.get_email_password()})
    try: agent.set_mode(cfg["mode"])
    except ValueError as e: raise HTTPException(400, str(e))
    logger.log("CONFIG_SAVE", {"username": cfg["username"]})
    return {"status": "saved"}

@app.post("/agent/mode")
async def set_mode(payload: ModePayload):
    if payload.mode not in VALID_MODES:
        raise HTTPException(400, "Invalid mode")
    needs_auth = (payload.mode == "active") or (agent._state.mode == "active")
    if needs_auth:
        if not payload.password or not auth.verify(payload.password):
            raise HTTPException(403, "Invalid password")
    else:
        # non-auth mode switch still requires valid session
        _require_session(payload.token)
    try:
        agent.set_mode(payload.mode)
        cfg = auth.load_config(_pw_from_token())
        cfg["mode"] = payload.mode
        auth.save_config(cfg, _pw_from_token())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "ok", "mode": payload.mode}

@app.post("/agent/stop")
async def stop_agent(payload: PasswordPayload):
    if not auth.verify(payload.password):
        raise HTTPException(403, "Invalid password")
    agent.stop()
    logger.log("STOP", {"source": "ui"})
    async def _exit():
        await asyncio.sleep(0.5)
        sys.exit(0)
    asyncio.create_task(_exit())
    return {"status": "stopping"}

@app.post("/agent/reset")
async def reset(payload: PasswordPayload):
    if not auth.verify(payload.password):
        raise HTTPException(403, "Invalid password")
    agent.stop()
    for p in ["data/model.pkl","data/scaler.pkl","data/score_dist.pkl",
              "data/training_data.npy","data/logs.json"]:
        try: Path(p).unlink()
        except Exception: pass
    auth.clear_credentials()
    import model as mdl
    with mdl._lock:
        mdl._state.clf = None; mdl._state.scaler = None; mdl._state.trained = False
    agent._state.__init__()
    agent.start()
    logger.log("RESET", {"source": "ui"})
    return {"status": "reset"}

@app.post("/agent/test-email")
async def test_email(payload: TestEmailPayload):
    cfg  = payload.dict()
    body = (f"Hello {payload.username},\n\nTest alert from KeyGuard.\n"
            f"Email configuration is working.\n\n— KeyGuard Security")
    ok = mailer.send_alert(cfg, "Test Alert — Configuration OK", body, alert_type="test", force=True)
    if ok: return {"status": "sent", "to": payload.email_recipient}
    raise HTTPException(502, "Email send failed — check credentials")


# ── startup: load config into agent ──────────────────────────────────────

_default_config = {
    "username":"user","mode":"training","email_sender":"","email_password":"",
    "email_recipient":"","alert_threshold":0.35,"alert_high":True,
    "alert_warn":False,"alert_retrain":False,"contamination":0.05,
}

def _load_config_with_pw(pw: str) -> dict:
    cfg = auth.load_config(pw)
    return {**_default_config, **cfg}

# Exported for run.py — password supplied after login
def init_agent_with_password(pw: str):
    global _session_pw
    _session_pw = pw
    cfg = _load_config_with_pw(pw)
    agent.set_config(cfg)
    try: agent.set_mode(cfg.get("mode", "training"))
    except ValueError: pass