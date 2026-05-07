import argparse, threading, time, webbrowser
from pathlib import Path
import uvicorn
import agent, logger
from server import app, _load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--host",       type=str, default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)

    cfg = _load_config()
    agent.set_config(cfg)
    agent.start()
    agent.set_mode(cfg.get("mode", "training"))

    url = f"http://{args.host}:{args.port}"
    logger.log("START", {"url": url, "mode": cfg.get("mode")})
    print(f"\n  KeyGuard → {url}\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()