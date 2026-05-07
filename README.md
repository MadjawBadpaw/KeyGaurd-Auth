# KeyGuard 🔐

> AI-powered keystroke biometrics authentication — knows *how* you type, not *what* you type.

KeyGuard builds a behavioral fingerprint from your typing rhythm. It silently monitors press/release timing and learns what normal looks like for you. When something feels off — a different user, an intruder, or a borrowed session — it notices, warns, and can lock down access automatically.

---

## What It Does

- **Learns your typing style** through an Isolation Forest anomaly-detection model trained on your keystroke timing patterns
- **Scores live sessions** in real time, comparing current typing against your enrolled baseline
- **Alerts on anomalies** via Gmail SMTP when confidence drops suspiciously
- **Triggers lockout** after repeated low-confidence detections
- **Protects sensitive actions** with a bcrypt-hashed dashboard password
- **Logs everything** — scores, events, and alert history — for review

---

## Privacy

KeyGuard is timing-only. It captures **when** keys are pressed and released — never which keys, never what you typed.

| Captured | Not captured |
|---|---|
| Press/release timestamps | Key identities |
| Inter-key timing (flight time) | Text content |
| Rhythm and pause patterns | Clipboard contents |

---

## How It Works

```
Keystrokes → Timing windows → Feature extraction → Anomaly model → Confidence score
                                                                          ↓
                                                              Alert / Warn / Lockout
```

Features extracted from each typing window:

| Feature | Description |
|---|---|
| **Dwell time** | How long each key is held |
| **Flight time** | Gap between key releases and presses |
| **Rhythm regularity** | Consistency of inter-key timing |
| **Speed profile** | Overall typing rate and variation |
| **Pause behavior** | Frequency and duration of pauses |
| **Error patterns** | Backspace and correction tendencies |
| **Entropy** | Randomness and unpredictability in the rhythm |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Windows (for `pynput` global key capture and the build script)
- A Gmail account with an [App Password](https://myaccount.google.com/apppasswords) configured (for email alerts)

### Installation

```bash
# Clone the repository
git clone https://github.com/MadjawBadpaw/KeyGaurd-Auth.git
cd KeyGaurd-Auth

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn numpy scikit-learn pynput bcrypt

# Launch
python run.py
```

Then open **http://127.0.0.1:8000** in your browser.

---

## Usage

### 1. First Launch — Set Your Password
On first run, you'll be prompted to create a dashboard password. This password gates all sensitive actions (enabling active mode, resetting the model, stopping the agent). It is stored as a bcrypt hash — never in plaintext.

### 2. Training Mode
Type naturally across any application. KeyGuard captures timing windows in the background and builds your behavioral baseline. The more samples collected, the more stable and accurate the model becomes. The dashboard shows your sample count and training progress.

### 3. Active Mode
Once sufficient samples are collected, switch to **Active Mode**. KeyGuard will begin scoring each new typing window against your baseline and displaying a live confidence score on the dashboard.

### 4. Anomaly Alerts
If a typing window scores below the configured confidence threshold, KeyGuard:
- Logs the anomaly event
- Optionally sends an email alert to the configured recipient
- Increments the consecutive-failure counter

### 5. Lockout Protection
After a configurable number of consecutive low-confidence windows, KeyGuard treats the session as compromised and triggers lockout behavior. This guards against an intruder who simply waits out a single warning.

### 6. Email Alerts
Configure the following in the dashboard:
- **Gmail sender address**
- **Gmail App Password** (not your account password)
- **Alert recipient address**

Use the **Test Alert** button to verify delivery before relying on it.

---

## Project Structure

```
KeyGaurd-Auth/
├── run.py              # Entry point — starts server and agent
├── server.py           # FastAPI routes and dashboard API
├── agent.py            # Background keystroke capture and scoring loop
├── features.py         # Timing feature extraction
├── model.py            # Isolation Forest model, training, and scoring
├── auth.py             # Password hashing and verification (bcrypt)
├── mailer.py           # Gmail SMTP alert delivery
├── logger.py           # Event and score logging
├── build.bat           # PyInstaller Windows build script
└── static/
    └── index.html      # Dashboard UI
```

---

## Building a Standalone Executable

```bat
build.bat
```

This packages KeyGuard into a single Windows executable using PyInstaller. The output is placed in `dist/`.

---

## Security Notes

- **Never commit `data/config.json`** — it contains your Gmail App Password and hashed dashboard password.
- **Use a strong, unique dashboard password.** It protects your lockout and reset controls.
- **Tune lockout thresholds carefully.** Too tight and you lock yourself out; too loose and the protection is ineffective. Start permissive, then tighten as you observe your own score variance.
- **Review logs regularly** during initial setup to understand your baseline score distribution.

### Recommended `.gitignore` additions

```
data/config.json
data/*.npy
data/*.pkl
data/logs.json
dist/
build/
__pycache__/
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| ML Model | scikit-learn (Isolation Forest) |
| Feature Engineering | NumPy |
| Key Capture | pynput |
| Auth | bcrypt |
| Alerts | Gmail SMTP |
| Frontend | HTML / CSS / JavaScript |
| Packaging | PyInstaller |

---

## Roadmap

- [ ] OS/session-level lockout (Windows lock screen integration)
- [ ] Multi-user enrollment support
- [ ] Per-session model calibration
- [ ] Smarter alert throttling to reduce noise
- [ ] Admin review dashboard with score history charts
- [ ] Replay and spoofing resistance testing

---

## Disclaimer

KeyGuard is an experimental research project in behavioral biometrics. It is not a replacement for enterprise-grade identity and access management. Treat it as an additional layer, not a standalone security solution.

---

## License

See [LICENSE](LICENSE) for details.
