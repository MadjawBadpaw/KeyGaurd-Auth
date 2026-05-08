"""
auth.py — Encrypted config + Windows Credential Manager for secrets.
- Guardian password hash stored in data/auth.bin (bcrypt, unencrypted — needed to verify before decrypt)
- config.json encrypted with AES-256-GCM, key derived from password via PBKDF2
- Email password stored in Windows Credential Manager via keyring
"""

import base64, bcrypt, json, os
from pathlib import Path
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import keyring

AUTH_BIN    = Path("data/auth.bin")
CONFIG_PATH = Path("data/config.enc")   # encrypted now, not .json
KEYRING_SVC = "KeyGuard"
KEYRING_USR = "email_password"
PBKDF2_ITER = 390_000
SALT_SIZE   = 32
NONCE_SIZE  = 12


# ── internal ────────────────────────────────────────────────────────────

def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITER,
    )
    return kdf.derive(password.encode())


def _encrypt(data: bytes, password: str) -> bytes:
    salt  = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key   = _derive_key(password, salt)
    ct    = AESGCM(key).encrypt(nonce, data, None)
    return salt + nonce + ct


def _decrypt(blob: bytes, password: str) -> bytes:
    salt  = blob[:SALT_SIZE]
    nonce = blob[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    ct    = blob[SALT_SIZE + NONCE_SIZE:]
    key   = _derive_key(password, salt)
    return AESGCM(key).decrypt(nonce, ct, None)


# ── public API ───────────────────────────────────────────────────────────

def is_set() -> bool:
    return AUTH_BIN.exists() and AUTH_BIN.stat().st_size > 0


def set_password(pw: str):
    """Hash password and store in auth.bin. Called once on first run."""
    AUTH_BIN.parent.mkdir(exist_ok=True)
    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt())
    AUTH_BIN.write_bytes(hashed)


def verify(pw: str) -> bool:
    try:
        stored = AUTH_BIN.read_bytes()
        return bcrypt.checkpw(pw.encode(), stored)
    except Exception:
        return False


def load_config(password: str) -> dict:
    """Decrypt and return config. Returns {} if no config yet."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        blob = CONFIG_PATH.read_bytes()
        raw  = _decrypt(blob, password)
        cfg  = json.loads(raw.decode())
        # pull email password from Credential Manager, not from file
        ep = keyring.get_password(KEYRING_SVC, KEYRING_USR)
        if ep:
            cfg["email_password"] = ep
        return cfg
    except Exception:
        return {}


def save_config(cfg: dict, password: str):
    """Encrypt and save config. Email password goes to Credential Manager."""
    cfg = dict(cfg)
    email_pw = cfg.pop("email_password", "")  # strip before encrypting
    if email_pw:
        keyring.set_password(KEYRING_SVC, KEYRING_USR, email_pw)

    CONFIG_PATH.parent.mkdir(exist_ok=True)
    blob = _encrypt(json.dumps(cfg).encode(), password)
    CONFIG_PATH.write_bytes(blob)


def get_email_password() -> str:
    return keyring.get_password(KEYRING_SVC, KEYRING_USR) or ""


def clear_credentials():
    """Called on reset — wipes encrypted config + credential manager entry."""
    try:
        CONFIG_PATH.unlink()
    except Exception:
        pass
    try:
        keyring.delete_password(KEYRING_SVC, KEYRING_USR)
    except Exception:
        pass