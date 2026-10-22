"""
Postgres persistence tests.

These run against a SEPARATE schema (`mcx_test`) so they can never touch real data,
and skip entirely when no server is reachable, so the suite stays runnable on a
machine without Postgres.

The most valuable test here is `test_raw_tape_is_sufficient_to_rebuild`: it replays
the stored raw tape through the full validation path and asserts the result matches
what was stored originally. That is what proves the raw tape really is enough to
rebuild everything downstream. If it were not, a corrected validation rule could
never be applied to history.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("psycopg")
import psycopg

from mcxrv import carry, ingest, liquidity, normalize
from mcxrv import signal as sig_mod
from mcxrv.db import DbConfig, repo
from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource

TEST_SCHEMA = "mcx_test"
START, END = date(2023, 1, 1), date(2023, 12, 31)


def _cfg() -> DbConfig:
    os.environ["MCX_DB_SCHEMA"] = TEST_SCHEMA
    return DbConfig.from_env()


def _server_available(cfg: DbConfig) -> bool:
    try:
        with psycopg.connect(cfg.conninfo("postgres"), connect_timeout=4):
            return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def db():
    prev = os.environ.get("MCX_DB_SCHEMA")
    cfg = _cfg()
    if not _server_available(cfg):
        if prev is None:
            os.environ.pop("MCX_DB_SCHEMA", None)
        else:
            os.environ["MCX_DB_SCHEMA"] = prev
        pytest.skip("no PostgreSQL server reachable; set MCX_DB_* or start one")
    repo.ensure_database(cfg)
    repo.drop_schema(cfg)
    repo.init_schema(cfg)
    try:
        yield cfg
    finally:
        repo.drop_schema(cfg)
        # Restore the caller's schema setting: a leaked MCX_DB_SCHEMA would point
        # later code at a scratch schema that this fixture has just dropped.
        if prev is None:
            os.environ.pop("MCX_DB_SCHEMA", None)
        else:
            os.environ["MCX_DB_SCHEMA"] = prev


@pytest.fixture(scope="module")
def loaded(db):
    """Load one year of synthetic data, and hand back what was computed in memory."""
    cfg = db
    src = SyntheticSource(SyntheticConfig(start=START, end=END, seed=21))
    raw = src.fetch_range(START, END)
    tidy, rep, _tc = ingest.build_dataset(raw)
    tidy = normalize.add_normalized(tidy)

    with repo.connect(cfg) as conn:
        run_id = repo.start_run(conn, "synthetic", START, END)
        n_raw = repo.write_raw(conn, raw, run_id, "synthetic")
        repo.upsert_contracts(conn, tidy)
        n_q = repo.upsert_quotes(conn, tidy, run_id)
        repo.write_dq(conn, run_id, rep)
        repo.finish_run(conn, run_id, n_raw, n_q)
        conn.commit()
    return {"cfg": cfg, "raw": raw, "tidy": tidy, "rep": rep,
            "run_id": run_id, "n_raw": n_raw}


# ------------------------------------------------------------------- schema

def test_schema_objects_exist(db):
    want = {"ingest_run", "bhavcopy_raw", "contract", "quote", "dq_flag",
            "carry_daily", "symbol_premium", "contract_residual",
            "symbol_residual", "pair_signal", "bt_trade", "bt_daily"}
    with repo.connect(db) as c:
        got = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (TEST_SCHEMA,)).fetchall()}
    assert want <= got


def test_views_exist(db):
    want = {"v_snapshot", "v_signal_board", "v_data_quality", "v_echo_audit",
            "v_latest_session"}
    with repo.connect(db) as c:
        got = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.views WHERE table_schema = %s",
            (TEST_SCHEMA,)).fetchall()}
    assert want <= got


def test_close_must_be_positive(loaded):
    """A non-positive settlement is unidentifiable and must not be storable."""
    with repo.connect(loaded["cfg"]) as c:
        with pytest.raises(psycopg.errors.CheckViolation):
            c.execute("INSERT INTO quote (contract_id, trade_date, symbol, "
                      "expiry_date, close) VALUES (%s, %s, %s, %s, %s)",
                      ("GOLDM|2023-03-03", date(2023, 2, 1), "GOLDM",
                       date(2023, 3, 3), -1))
        c.rollback()


# ------------------------------------------------------------------ the tape

def test_requested_date_is_preserved(loaded):
    """
    Without the requested date the echo is undetectable after the fact. This column
    is the reason the raw tape is worth keeping at all.
    """
    with repo.connect(loaded["cfg"]) as c:
        n = c.execute("SELECT count(*) FROM bhavcopy_raw "
                      "WHERE requested_date IS NOT NULL").fetchone()[0]
    assert n > 0


def test_echo_audit_view_finds_echoes(loaded):
    with repo.connect(loaded["cfg"]) as c:
        audit = repo.echo_audit(c)
    assert not audit.empty, "synthetic data injects echoes; the view must see them"
    assert (audit["requested_date"] != pd.to_datetime(
        audit["raw_date"], format="%m/%d/%Y").dt.date).all()


def test_raw_tape_is_sufficient_to_rebuild(loaded):
    """
    Replay stored raw through the identical validation path and compare with what was
    stored. Proves history can be rebuilt under a corrected rule without re-fetching.
    """
    cfg = loaded["cfg"]
    with repo.connect(cfg) as c:
        raw_back = repo.read_raw(c, start=START, end=END)
    assert len(raw_back) == loaded["n_raw"]

    rebuilt, _rep, _tc = ingest.build_dataset(raw_back)
    stored = loaded["tidy"]

    a = rebuilt[["date", "contract_id", "close", "volume"]].sort_values(
        ["date", "contract_id"]).reset_index(drop=True)
    b = stored[["date", "contract_id", "close", "volume"]].sort_values(
        ["date", "contract_id"]).reset_index(drop=True)
    assert len(a) == len(b)
    assert (a["contract_id"].to_numpy() == b["contract_id"].to_numpy()).all()
    assert np.allclose(a["close"].astype(float), b["close"].astype(float))


# ------------------------------------------------------------- idempotency

def test_reload_converges_rather_than_accumulates(loaded):
    """
    The property that makes a daily append safe to re-run. Derived tables upsert;
    only the raw tape appends, because a duplicate upstream row is part of what
    happened and belongs in the record.
    """
    cfg = loaded["cfg"]
    with repo.connect(cfg) as c:
        before_q = c.execute("SELECT count(*) FROM quote").fetchone()[0]
        before_c = c.execute("SELECT count(*) FROM contract").fetchone()[0]
        before_r = c.execute("SELECT count(*) FROM bhavcopy_raw").fetchone()[0]

    tidy = loaded["tidy"]
    with repo.connect(cfg) as c:
        rid = repo.start_run(c, "synthetic", START, END)
        repo.write_raw(c, loaded["raw"], rid, "synthetic")
        repo.upsert_contracts(c, tidy)
        repo.upsert_quotes(c, tidy, rid)
        c.commit()

    with repo.connect(cfg) as c:
        after_q = c.execute("SELECT count(*) FROM quote").fetchone()[0]
        after_c = c.execute("SELECT count(*) FROM contract").fetchone()[0]
        after_r = c.execute("SELECT count(*) FROM bhavcopy_raw").fetchone()[0]

    assert after_q == before_q, "quotes must upsert, not duplicate"
    assert after_c == before_c, "contracts must upsert, not duplicate"
    assert after_r == before_r * 2, "raw tape is append-only by design"


def test_quote_primary_key_blocks_duplicate_contract_day(loaded):
    with repo.connect(loaded["cfg"]) as c:
        row = c.execute("SELECT contract_id, trade_date, symbol, expiry_date, close "
                        "FROM quote LIMIT 1").fetchone()
        with pytest.raises(psycopg.errors.UniqueViolation):
            c.execute("INSERT INTO quote (contract_id, trade_date, symbol, "
                      "expiry_date, close) VALUES (%s, %s, %s, %s, %s)", row)
        c.rollback()


# ----------------------------------------------------------------- round trip

def test_quotes_round_trip_preserves_values_and_flags(loaded):
    cfg = loaded["cfg"]
    with repo.connect(cfg) as c:
        back = repo.read_quotes(c, start=START, end=END)
    stored = loaded["tidy"]
    assert len(back) == len(stored)

    a = back.sort_values(["date", "contract_id"]).reset_index(drop=True)
    b = stored.sort_values(["date", "contract_id"]).reset_index(drop=True)
    assert np.allclose(a["close"].astype(float), b["close"].astype(float))
    assert np.allclose(a["norm_close"].astype(float),
                       b["norm_close"].astype(float), rtol=1e-9)
    assert (a["no_trade"].to_numpy() == b["no_trade"].to_numpy()).all()
    assert (a["settle_untradeable"].to_numpy()
            == b["settle_untradeable"].to_numpy()).all()
    assert int(a["date_echo"].sum()) == int(b["date_echo"].sum())


def test_model_outputs_persist_and_handle_non_finite(loaded):
    """
    Infinity is a legitimate half-life ("does not revert") but not a storable double.
    It must land as NULL with the gate column carrying the meaning, not crash or
    silently become a large finite number.
    """
    cfg = loaded["cfg"]
    with repo.connect(cfg) as c:
        q = repo.read_quotes(c, start=START, end=END)
    q = normalize.add_normalized(q)
    rows, daily, prem = carry.fit(q, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    sr = carry.symbol_residual(rows)
    pairs = sig_mod.pair_spreads(sr, sig_mod.SignalConfig(min_history=40,
                                                         lookback=120))

    # force an infinite half-life through the writer
    if not pairs.empty:
        pairs = pairs.copy()
        pairs.loc[pairs.index[:5], "halflife"] = np.inf

    with repo.connect(cfg) as c:
        n = repo.write_model(c, daily=daily, prem=prem, rows=rows,
                             sym_resid=sr, pairs=pairs)
        c.commit()
        assert n.get("carry_daily", 0) > 0
        assert n.get("contract_residual", 0) > 0
        if not pairs.empty:
            nulls = c.execute("SELECT count(*) FROM pair_signal "
                              "WHERE halflife IS NULL").fetchone()[0]
            assert nulls >= 5


def test_snapshot_view_picks_a_tradable_contract(loaded):
    cfg = loaded["cfg"]
    with repo.connect(cfg) as c:
        snap = repo.snapshot(c)
    assert not snap.empty
    assert snap["symbol"].is_unique, "one row per symbol"
    assert (snap["norm_close"].astype(float) > 0).all()


def test_signal_board_names_the_binding_gate(loaded):
    cfg = loaded["cfg"]
    with repo.connect(cfg) as c:
        board = repo.signal_board(c)
    if board.empty:
        pytest.skip("no signals stored for the latest session")
    blocked = board[~board["signal_ok"].fillna(False)]
    assert blocked["binding_gate"].notna().all(), \
        "a blocked pair must say which gate blocked it"


def test_anomaly_queue_persists_and_stays_advisory(loaded):
    """
    The anomaly table round-trips, and `v_anomaly_review` surfaces only the rows no
    rule caught -- the ones actually worth a human's attention.
    """
    pytest.importorskip("sklearn")
    from mcxrv import anomaly, liquidity

    cfg = loaded["cfg"]
    with repo.connect(cfg) as c:
        q = repo.read_quotes(c, start=START, end=END)
    q = normalize.add_normalized(q)
    rows, daily, _prem = carry.fit(q, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())

    det = anomaly.detect(rows, anomaly.AnomalyConfig(contamination=0.02))
    if det.empty:
        pytest.skip("detector produced nothing on this sample")

    with repo.connect(cfg) as c:
        n = repo.write_anomalies(c, det)
        c.commit()
        assert n == len(det)

        # idempotent like every other derived table
        repo.write_anomalies(c, det)
        c.commit()
        total = c.execute("SELECT count(*) FROM dq_anomaly").fetchone()[0]
        assert total == len(det)

        review = repo.anomaly_review(c, limit=100)
        if not review.empty:
            assert {"score", "top_feature", "close"} <= set(review.columns)
            assert review["score"].is_monotonic_increasing, "worst first"

        # features survived as jsonb
        row = c.execute("SELECT features FROM dq_anomaly "
                        "WHERE features IS NOT NULL LIMIT 1").fetchone()
        assert row and isinstance(row[0], dict)


def test_dq_ledger_written(loaded):
    with repo.connect(loaded["cfg"]) as c:
        df = repo._fetch_df(c, "SELECT issue, rows FROM dq_flag WHERE run_id = %s",
                            [loaded["run_id"]])
    assert not df.empty
    assert "date_echo_rows" in set(df["issue"])
