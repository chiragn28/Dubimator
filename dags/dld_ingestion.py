import os
from datetime import UTC, datetime
from pathlib import Path

from airflow.decorators import dag, task


@dag(
    dag_id="dld_ingestion",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    tags=["dubimator"],
)
def dld_ingestion():
    @task
    def ingest() -> dict:
        # Imported inside the task so DAG parsing doesn't pay for importing Polars.
        from ingestion.config import DbSettings
        from ingestion.pipeline import run_pipeline

        summary = run_pipeline(Path(os.environ["DLD_CSV_PATH"]), DbSettings.from_env())
        return {
            "run_id": summary.run_id,
            "rows_read": summary.rows_read,
            "rows_market_sale": summary.rows_market_sale,
        }

    ingest()


dld_ingestion()
