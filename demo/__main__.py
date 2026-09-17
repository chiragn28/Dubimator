"""Command line: python -m demo [--port 8501] [--host 127.0.0.1] — starts the Streamlit app.

Binds to localhost by default; pass `--host 0.0.0.0` only to expose it on the network.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

APP = Path(__file__).resolve().parent / "app.py"


DEFAULT_HOST = "127.0.0.1"


def build_command(port: int, host: str = DEFAULT_HOST) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(port), "--server.address", host,
        "--server.headless", "true",
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m demo", description="Start the Streamlit demo.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"address to bind (default {DEFAULT_HOST})"
    )
    args = parser.parse_args(argv)
    load_dotenv()  # DEMO_API_URL / DEMO_API_KEY come from .env
    return subprocess.run(build_command(args.port, args.host), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
