"""
run.py — Entry point. Prompts for password, decrypts config, starts server.
"""

import argparse, getpass, sys, threading, webbrowser
from pathlib import Path

import uvicorn
import agent, logger, auth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--host",       type=str, default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)

    # First run: set password
    if not auth.is_set():
        print("\n  KeyGuard — First Run Setup")
        print("  ─────────────────────────────")
        while True:
            pw  = getpass.getpass("  Set password (min 6 chars): ")
            pw2 = getpass.getpass("  Confirm password: ")
            if pw != pw2:
                print("  Passwords don't match, try again.\n")
                continue
            if len(pw) < 6:
                print("  Too short.\n")
                continue
            auth.set_password(pw)
            print("  Password set.\n")
            break
    else:
        pw = getpass.getpass("\n  KeyGuard — Enter password: ")
        if not auth.verify(pw):
            print("  Wrong password. Exiting.")
            sys.exit(1)

    # Load config + start agent
    from server import app, init_agent_with_password
    agent.start()
    init_agent_with_password(pw)

    url = f"http://{args.host}:{args.port}"
    logger.log("START", {"url": url})
    print(f"  KeyGuard → {url}\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()