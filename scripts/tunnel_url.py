"""Print the Cloudflare quick-tunnel URL from the cloudflared container's logs.

Usage: uv run python scripts/tunnel_url.py [--service cloudflared-quick]
Exits 1 (message on stderr) when compose fails or no URL has been logged yet.
"""

import argparse
import re
import subprocess
import sys

URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com(?![\w.-])")


def extract_url(text: str) -> str | None:
    match = URL_PATTERN.search(text)
    return match.group(0) if match else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--service", default="cloudflared-quick", help="compose service name")
    args = parser.parse_args(argv)
    cmd = ["docker", "compose", "logs", "--no-color", args.service]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"could not run docker compose logs: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(
            f"docker compose logs {args.service} failed: {result.stderr.strip()}", file=sys.stderr
        )
        return 1
    url = extract_url(result.stdout + result.stderr)
    if url is None:
        print(
            f"no trycloudflare.com URL in the {args.service} logs yet "
            "(named-tunnel mode has none; otherwise wait a few seconds and retry)",
            file=sys.stderr,
        )
        return 1
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
