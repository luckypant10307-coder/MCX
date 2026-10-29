"""
Database connection configuration.

Credentials come from the environment, never from source. A `.env` file at the repo
root is read if present (and is gitignored); explicit environment variables win over
it, so a CI or production setting cannot be silently overridden by a stray file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULTS = {
    "MCX_DB_HOST": "localhost",
    "MCX_DB_PORT": "5432",
    "MCX_DB_NAME": "mcx_gold",
    "MCX_DB_USER": "postgres",
    "MCX_DB_PASSWORD": "postgres",
    "MCX_DB_SCHEMA": "mcx",
}

_ENV_LOADED = False


def load_dotenv(path: str | None = None) -> None:
    """
    Read a simple KEY=VALUE .env into os.environ WITHOUT clobbering what is already set.

    Deliberately minimal: no interpolation, no export syntax, no dependency. Anything
    more belongs in a real secrets manager, not in a repo helper.
    """
    global _ENV_LOADED
    if _ENV_LOADED and path is None:
        return
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, ".env")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)   # existing env wins
    _ENV_LOADED = True


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    schema: str

    @classmethod
    def from_env(cls) -> "DbConfig":
        load_dotenv()
        g = lambda k: os.environ.get(k, DEFAULTS[k])
        return cls(host=g("MCX_DB_HOST"), port=int(g("MCX_DB_PORT")),
                   dbname=g("MCX_DB_NAME"), user=g("MCX_DB_USER"),
                   password=g("MCX_DB_PASSWORD"), schema=g("MCX_DB_SCHEMA"))

    def conninfo(self, dbname: str | None = None) -> str:
        return ("host={} port={} dbname={} user={} password={}"
                .format(self.host, self.port, dbname or self.dbname,
                        self.user, self.password))

    def safe_repr(self) -> str:
        """Connection description with the password removed, for logs."""
        return "postgresql://{}@{}:{}/{} (schema {})".format(
            self.user, self.host, self.port, self.dbname, self.schema)
