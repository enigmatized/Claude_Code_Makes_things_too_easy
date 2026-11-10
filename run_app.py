"""Start the local UI: python3 run_app.py [--port 8501]."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--mode", choices=["research", "deploy"], default="research")
    parser.add_argument("--published-root", type=Path, help="Self-contained published-results directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    env = dict(os.environ, MLFACTOR_APP_MODE=args.mode)
    if args.mode == "deploy":
        env["MLFACTOR_READ_ONLY"] = "1"
    if args.published_root:
        env["MLFACTOR_PUBLISHED_ROOT"] = str(args.published_root.resolve())
    try:
        return subprocess.call([
            sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
            "--server.address", "127.0.0.1", "--server.port", str(args.port),
            "--server.headless", "true", "--browser.gatherUsageStats", "false",
        ], cwd=root, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
