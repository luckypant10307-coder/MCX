"""
Create the database and schema.

Usage:
  python scripts/db_init.py              # create if absent, apply schema
  python scripts/db_init.py --reset      # DROP the schema first (destructive)
  python scripts/db_init.py --check      # connectivity and row counts only
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg

from mcxrv.db import DbConfig, repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="drop the schema before recreating it (destructive)")
    ap.add_argument("--check", action="store_true",
                    help="report connectivity and row counts, change nothing")
    args = ap.parse_args()

    cfg = DbConfig.from_env()
    print("target: {}".format(cfg.safe_repr()))

    try:
        with psycopg.connect(cfg.conninfo("postgres"), connect_timeout=6) as c:
            ver = c.execute("SELECT version()").fetchone()[0]
        print("server : {}".format(ver.split(",")[0]))
    except psycopg.Error as e:
        print("\nCannot reach PostgreSQL: {}".format(str(e).strip()))
        print("Check the server is running and that .env matches your setup.")
        print("Settings come from .env or the environment (see .env.example).")
        raise SystemExit(2)

    if args.check:
        try:
            with repo.connect(cfg) as c:
                print("\nrow counts:")
                print(repo.table_counts(c).to_string(index=False))
        except psycopg.Error as e:
            print("schema not initialised yet ({})".format(str(e).strip()[:80]))
        return

    if args.reset:
        confirm = os.environ.get("MCX_DB_CONFIRM_RESET", "")
        if confirm != "yes":
            print("\n--reset drops schema '{}' and everything in it."
                  .format(cfg.schema))
            print("Re-run with MCX_DB_CONFIRM_RESET=yes to proceed.")
            raise SystemExit(1)
        repo.drop_schema(cfg)
        print("dropped schema {}".format(cfg.schema))

    created = repo.ensure_database(cfg)
    print("database: {}".format("created" if created else "already present"))

    repo.init_schema(cfg)
    print("schema  : applied")

    with repo.connect(cfg) as c:
        print("\nrow counts:")
        print(repo.table_counts(c).to_string(index=False))


if __name__ == "__main__":
    main()
