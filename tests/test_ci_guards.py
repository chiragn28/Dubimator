"""CI infra guards (Phase 9): the unset-port skip, the REQUIRE_INFRA fail, and marker
registration. Exercises `conftest.pg_test_db` and `conftest._infra_unavailable` directly with
monkeypatch, rather than needing a real Postgres — that's what tests/test_infra_smoke.py and the
`pg_test_db`-using suites are for.

Fetches the *already loaded* tests/conftest.py plugin object from pytest's plugin manager
instead of `import conftest` — several subdirectories (tests/ingestion, tests/listings,
tests/search, ...) have their own conftest.py, and since none of them are packages (no
__init__.py), a bare `import conftest` risks resolving to whichever one pytest happened to
import first under that same top-level module name.
"""

import pathlib

import pytest

_ROOT_CONFTEST_PATH = pathlib.Path(__file__).resolve().parent / "conftest.py"


@pytest.fixture
def conftest(pytestconfig):
    for plugin in pytestconfig.pluginmanager.get_plugins():
        plugin_file = getattr(plugin, "__file__", None)
        if plugin_file and pathlib.Path(plugin_file).resolve() == _ROOT_CONFTEST_PATH:
            return plugin
    raise RuntimeError(f"tests/conftest.py not found among loaded plugins ({_ROOT_CONFTEST_PATH})")


def test_pg_test_db_skips_when_port_unset(monkeypatch, conftest):
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("REQUIRE_INFRA", raising=False)
    gen = conftest.pg_test_db.__wrapped__()
    with pytest.raises(pytest.skip.Exception, match="POSTGRES_PORT is not set"):
        next(gen)


def test_pg_test_db_fails_when_require_infra_and_port_unset(monkeypatch, conftest):
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.setenv("REQUIRE_INFRA", "1")
    gen = conftest.pg_test_db.__wrapped__()
    with pytest.raises(pytest.fail.Exception, match="POSTGRES_PORT is not set"):
        next(gen)


def test_pg_test_db_fails_when_require_infra_and_port_unreachable(monkeypatch, conftest):
    # Port 1 needs a privileged process and nothing listens on it in CI or locally.
    monkeypatch.setenv("POSTGRES_PORT", "1")
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("REQUIRE_INFRA", "1")
    gen = conftest.pg_test_db.__wrapped__()
    with pytest.raises(pytest.fail.Exception, match="not reachable"):
        next(gen)


def test_infra_unavailable_skips_by_default(monkeypatch, conftest):
    monkeypatch.delenv("REQUIRE_INFRA", raising=False)
    with pytest.raises(pytest.skip.Exception, match="boom"):
        conftest._infra_unavailable("boom")


def test_infra_unavailable_fails_under_require_infra(monkeypatch, conftest):
    monkeypatch.setenv("REQUIRE_INFRA", "1")
    with pytest.raises(pytest.fail.Exception, match="boom"):
        conftest._infra_unavailable("boom")


def test_gpu_and_live_and_db_markers_registered(pytestconfig):
    names = {line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")}
    assert {"gpu", "live", "db"} <= names


def test_pg_test_db_is_auto_marked_db(conftest):
    # Exercises the pytest_collection_modifyitems hook directly against a stand-in "item",
    # rather than depending on some other test file continuing to use pg_test_db. The hook is
    # also exercised end-to-end whenever the real suite runs under `-m db` / `-m "not db"`.
    class FakeItem:
        fixturenames = ("pg_test_db",)

        def __init__(self):
            self.markers_added = []

        def add_marker(self, marker):
            self.markers_added.append(marker)

    item = FakeItem()
    conftest.pytest_collection_modifyitems([item])
    assert any(m.name == "db" for m in item.markers_added)
