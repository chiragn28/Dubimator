"""Build the static architecture site for Vercel: site/index.html from docs/architecture.html.

docs/architecture.html holds only the page body (the artifact host wraps it), so this adds the
document shell a static host needs. Deploy with: `vercel deploy site --prod`.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "architecture.html"
SITE = REPO / "site"

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Dubimator: a Dubai real-estate ML platform — build status, architecture and results.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏗️</text></svg>">
{head}
</head>
<body>
{body}
</body>
</html>
"""


def build(source: Path = SOURCE, site: Path = SITE) -> Path:
    page = source.read_text(encoding="utf-8")
    # <title>, font links and <style> go in <head>; everything from the first <div> is the body.
    split = page.index("<div")
    head, body = page[:split].strip(), page[split:].strip()
    if re.search(r"<(html|head|body)\b", page, re.IGNORECASE):
        raise ValueError(f"{source} already has a document shell")
    site.mkdir(parents=True, exist_ok=True)
    target = site / "index.html"
    target.write_text(SHELL.format(head=head, body=body), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(f"wrote {build()}")
    sys.exit(0)
