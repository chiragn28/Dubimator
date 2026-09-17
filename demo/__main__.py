"""Command line: python -m demo [--port 8501] — starts the Streamlit app."""

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

APP = Path(__file__).resolve().parent / "app.py"


def build_command(port: int) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(port), "--server.headless", "true",
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m demo", description="Start the Streamlit demo.")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)
    load_dotenv()  # DEMO_API_URL / DEMO_API_KEY come from .env
    return subprocess.run(build_command(args.port), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
