# KeyGuard

AI-powered typing biometrics authentication with real-time confidence scoring, anomaly alerts, and lockout protection.

KeyGuard learns how a user types by analyzing keystroke timing patterns only. It does not store typed text or key content. After enough training data is collected, it scores live typing windows against the learned baseline, sends alerts on suspicious behavior, and can lock out access when confidence repeatedly drops below the accepted threshold.

## Features

- Keystroke dynamics authentication
- Privacy-first timing capture only
- Real-time confidence scoring
- Training mode and active protection mode
- Low-confidence anomaly detection
- Email alerts through Gmail SMTP
- Lockout after repeated failed confidence checks
- Password-protected dashboard actions
- Retraining as more samples are collected
- Event logging and score history
- Desktop packaging with PyInstaller

## How It Works

KeyGuard captures typing behavior in timed windows and extracts statistical features such as:

- Dwell timing
- Flight timing
- Rhythm regularity
- Speed profile
- Pause behavior
- Error and correction patterns
- Entropy and consistency features

These features are passed into an anomaly-detection model that estimates whether the current typing pattern matches the enrolled user.

When confidence stays healthy, access remains normal. When confidence becomes suspicious, KeyGuard can warn, alert, and eventually trigger lockout protection after repeated low-confidence detections.

## Tech Stack

- Python
- FastAPI
- Uvicorn
- NumPy
- scikit-learn
- pynput
- bcrypt
- HTML/CSS/JavaScript

## Project Structure

```text
.
├── agent.py            # Background keystroke capture and scoring loop
├── auth.py             # Password hashing and verification
├── build.bat           # Windows build script
├── features.py         # Keystroke feature extraction
├── logger.py           # Event logging
├── mailer.py           # Gmail SMTP alerts
├── model.py            # Isolation Forest model and scoring
├── run.py              # App entry point
├── server.py           # FastAPI backend
└── static/
    └── index.html      # Dashboard UI
Installation
Clone the repo
git clone https://github.com/MadjawBadpaw/KeyGaurd-Auth.git
cd KeyGaurd-Auth
Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate
Install dependencies
pip install fastapi uvicorn numpy scikit-learn pynput bcrypt
Run
python run.py
Open:

http://127.0.0.1:8000
Usage
1. First launch
Set a dashboard password
This password protects sensitive actions like enabling active mode, reset, stop, and protected access flows
2. Training mode
Type naturally in any app
KeyGuard collects timing windows and builds your typing baseline
More samples improve reliability and stability
3. Active mode
Once enough samples are collected, switch to active mode
KeyGuard begins scoring live typing against your baseline
Low-confidence behavior can trigger warnings, alerts, and lockout
4. Lockout protection
If confidence repeatedly falls below the configured threshold, KeyGuard can treat the session as suspicious
Repeated anomaly streaks trigger lockout behavior
This adds another layer of protection beyond one-time alerts
5. Email alerts
Configure Gmail sender, recipient, and App Password
Test alert delivery from the dashboard
Alerts can be sent for anomalies, warnings, and retrains
Privacy
KeyGuard is designed to capture timing information only.

It does:

Store press/release timing
Build feature vectors from typing rhythm
It does not:

Store typed characters
Store text content
Store clipboard contents
Build Executable
build.bat
This creates a packaged Windows executable using PyInstaller.

Security Notes
Keep data/config.json out of Git
Never commit Gmail App Passwords
Use a strong dashboard password
Review logs regularly during testing
Lockout thresholds should be tuned carefully to reduce false positives
Git Ignore
Recommended files to keep out of Git:

data/config.json
data/*.npy
data/*.pkl
data/logs.json
dist/
build/
__pycache__/
Future Improvements
Stronger OS/session-level lockout integrations
Better multi-user enrollment
Model calibration per session
More advanced alert throttling
Admin review dashboard
Replay/spoof resistance testing
Disclaimer
KeyGuard is an experimental behavioral authentication project and should not be treated as a complete replacement for enterprise-grade identity security.

License
MIT
