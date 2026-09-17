import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_site.py"
spec = importlib.util.spec_from_file_location("build_site", SCRIPT)
build_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_site)


def test_build_wraps_the_page_in_a_full_document(tmp_path):
    source = tmp_path / "architecture.html"
    source.write_text(
        '<title>Dubimator Architecture</title>\n<style>body{}</style>\n\n<div class="wrap">hi</div>\n',
        encoding="utf-8",
    )
    target = build_site.build(source, tmp_path / "site")
    html = target.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    head, body = html.split("<body>")
    assert "<title>Dubimator Architecture</title>" in head and "<style>" in head
    assert '<div class="wrap">hi</div>' in body
    assert 'name="viewport"' in head


def test_build_rejects_a_page_that_already_has_a_shell(tmp_path):
    source = tmp_path / "architecture.html"
    source.write_text("<html><body><div>x</div></body></html>", encoding="utf-8")
    with pytest.raises(ValueError):
        build_site.build(source, tmp_path / "site")


def test_the_real_architecture_page_builds(tmp_path):
    html = build_site.build(site=tmp_path).read_text(encoding="utf-8")
    assert "<title>Dubimator Architecture</title>" in html
    assert html.count("<body>") == 1
