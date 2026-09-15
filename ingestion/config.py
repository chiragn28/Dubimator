import os
from dataclasses import dataclass

import psycopg2


@dataclass(frozen=True)
class DbSettings:
    host: str
    port: int
    user: str
    password: str
    dbname: str

    @classmethod
    def from_env(cls) -> "DbSettings":
        # No default port: a script that forgot load_dotenv() would otherwise silently connect
        # to whatever Postgres owns 5432, which need not be this project's.
        port = os.environ.get("POSTGRES_PORT")
        if port is None:
            raise RuntimeError(
                "POSTGRES_PORT is not set: call load_dotenv() before DbSettings.from_env() so "
                "the project's .env is read (or set POSTGRES_PORT in the environment)"
            )
        return cls(
            host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
            port=int(port),
            user=os.environ.get("POSTGRES_USER", "zestimator"),
            password=os.environ.get("POSTGRES_PASSWORD", "changeme"),
            dbname=os.environ.get("POSTGRES_DB", "zestimator"),
        )

    def connect(self):
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname,
            connect_timeout=10,
        )
