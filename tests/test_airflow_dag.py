import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PROBE = (
    "import json; import ingestion.pipeline; from airflow.models import DagBag; "
    "bag = DagBag(include_examples=False); "
    "print('RESULT ' + json.dumps({'errors': {k: str(v) for k, v in bag.import_errors.items()}, "
    "'dags': sorted(bag.dag_ids)}))"
)


def _airflow_running() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "ps", "--status", "running", "--services"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0 and "airflow" in result.stdout.split()


def test_dld_ingestion_dag_loads_in_airflow():
    if not _airflow_running():
        pytest.skip(
            "Airflow container is not running — start the stack with `docker compose up -d --wait`."
        )
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "airflow", "python", "-c", PROBE],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
    probe = json.loads(line.removeprefix("RESULT "))
    assert probe["errors"] == {}
    assert "dld_ingestion" in probe["dags"]
