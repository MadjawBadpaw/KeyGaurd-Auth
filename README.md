# KeyGuard

AI-powered typing biometrics authentication with real-time confidence scoring, anomaly alerts, and lockout protection.

KeyGuard learns how a user types by analyzing keystroke timing patterns only. It does not store typed text or key content. After enough training data is collected, it scores live typing windows against the learned baseline, sends alerts on suspicious behavior, and can lock out access when confidence repeatedly drops below the accepted threshold.

---

## Features

- Keystroke dynamics authentication
- Privacy-first timing capture only — no keystrokes stored
- Real-time confidence scoring
- Training mode and active protection mode
- Low-confidence anomaly detection
- Email alerts through Gmail SMTP
- Lockout after repeated failed confidence checks
- Password-protected dashboard actions
- Retraining as more samples are collected
- Event logging and score history
- Desktop packaging with PyInstaller

---

## How It Works

KeyGuard captures typing behavior in timed windows and extracts statistical features:

| Signal | Description |
|---|---|
| Dwell timing | How long each key is held |
| Flight timing | Time between key releases and presses |
| Rhythm regularity | Variance in inter-key intervals |
| Speed profile | Words-per-minute over time |
| Pause behavior | Where and how often pauses occur |
| Error patterns | Correction and backspace cadence |
| Entropy | Consistency across a typing window |

These features are passed into an Isolation Forest anomaly-detection model that estimates whether the current typing pattern matches the enrolled user.

When confidence stays healthy, access remains normal. When confidence becomes suspicious, KeyGuard can warn, alert, and eventually trigger lockout protection after repeated low-confidence detections.

---

## Tech Stack

- Python
- FastAPI
- Uvicorn
- NumPy
- scikit-learn
- pynput
- bcrypt
- HTML / CSS / JavaScript

---

## Project Structure

```
.
├── agent.py        # Background keystroke capture and scoring loop
├── auth.py         # Password hashing and verification
├── build.bat       # Windows build script
├── features.py     # Keystroke feature extraction
├── logger.py       # Event logging
├── mailer.py       # Gmail SMTP alerts
├── model.py        # Isolation Forest model and scoring
├── run.py          # App entry point
├── server.py       # FastAPI backend
└── static/
    └── index.html  # Dashboard UI
```
