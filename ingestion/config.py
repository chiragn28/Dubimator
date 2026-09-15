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
        return cls(
            host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
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
