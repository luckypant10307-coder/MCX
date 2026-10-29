"""
Read and write the Postgres store.

Two write paths, deliberately different:

  raw tape   COPY, append-only. Fast and immutable. Never updated, never deduped --
             duplicates in the upstream are part of what happened and belong in the
             record. Deduplication is a property of the DERIVED tables.

  derived    upsert on the natural key. Idempotent, so re-running a day or reloading
             a whole range converges rather than accumulating. This is what makes a
             daily append safe to run twice.

NaN handling is explicit throughout: pandas uses NaN for missing floats, Postgres
wants NULL, and a NaN silently written into a numeric column becomes the string 'NaN'
in some drivers and an error in others. `_clean` converts once, at the boundary.
"""
from __future__ import annotations

import contextlib
import io
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from .config import DbConfig

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_schema.sql")


# ---------------------------------------------------------------- connection

def connect(cfg: DbConfig | None = None, dbname: str | None = None,
            autocommit: bool = False) -> psycopg.Connection:
    cfg = cfg or DbConfig.from_env()
    conn = psycopg.connect(cfg.conninfo(dbname), autocommit=autocommit)
    if dbname is None:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET search_path TO {}, public")
                        .format(sql.Identifier(cfg.schema)))
        if not autocommit:
            conn.commit()
    return conn


def ensure_database(cfg: DbConfig | None = None) -> bool:
    """Create the database if absent. Returns True if it was created."""
    cfg = cfg or DbConfig.from_env()
    with psycopg.connect(cfg.conninfo("postgres"), autocommit=True) as c:
        exists = c.execute("SELECT 1 FROM pg_database WHERE datname = %s",
                           (cfg.dbname,)).fetchone()
        if exists:
            return False
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(cfg.dbname)))
        return True


def init_schema(cfg: DbConfig | None = None) -> None:
    """
    Create the configured schema and apply the DDL inside it.

    The schema name lives here rather than in schema.sql so that MCX_DB_SCHEMA is
    honoured -- which is what lets the test suite isolate itself in a scratch schema
    instead of writing over real data.
    """
    cfg = cfg or DbConfig.from_env()
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        ddl = fh.read()
    with psycopg.connect(cfg.conninfo()) as c:
        with c.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}")
                        .format(sql.Identifier(cfg.schema)))
            cur.execute(sql.SQL("SET search_path TO {}, public")
                        .format(sql.Identifier(cfg.schema)))
            cur.execute(ddl)
        c.commit()


def drop_schema(cfg: DbConfig | None = None) -> None:
    """Destructive. Used by tests and by an explicit --reset."""
    cfg = cfg or DbConfig.from_env()
    with connect(cfg, dbname=cfg.dbname, autocommit=True) as c:
        c.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE")
                  .format(sql.Identifier(cfg.schema)))


# ------------------------------------------------------------------- helpers

def _clean(v):
    """pandas/numpy value -> something psycopg can bind."""
    if v is None:
        return None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if not np.isfinite(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.date()
    if v is pd.NaT:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _rows(df: pd.DataFrame, cols) -> list:
    sub = df.reindex(columns=cols)
    return [tuple(_clean(v) for v in rec) for rec in sub.itertuples(index=False)]


def _upsert(conn, table: str, cols, keys, df: pd.DataFrame,
            update: bool = True, page: int = 5000) -> int:
    """Insert with ON CONFLICT DO UPDATE on `keys`. Returns rows sent."""
    if df is None or df.empty:
        return 0
    data = _rows(df, cols)
    if not data:
        return 0
    upd = [c for c in cols if c not in keys]
    ident = lambda xs: sql.SQL(", ").join(sql.Identifier(x) for x in xs)
    if update and upd:
        action = sql.SQL("DO UPDATE SET {}").format(
            sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
                for c in upd))
    else:
        action = sql.SQL("DO NOTHING")
    stmt = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) {}").format(
        sql.Identifier(table), ident(cols),
        sql.SQL(", ").join(sql.Placeholder() * len(cols)),
        ident(keys), action)
    n = 0
    with conn.cursor() as cur:
        for i in range(0, len(data), page):
            cur.executemany(stmt, data[i:i + page])
            n += len(data[i:i + page])
    return n


# ------------------------------------------------------------------ run log

def start_run(conn, source: str, start: date | None, end: date | None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingest_run (source, range_start, range_end) "
            "VALUES (%s, %s, %s) RETURNING run_id", (source, start, end))
        return cur.fetchone()[0]


def finish_run(conn, run_id: int, rows_fetched: int, rows_validated: int,
               status: str = "ok", note: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingest_run SET finished_at = now(), rows_fetched = %s, "
            "rows_validated = %s, status = %s, note = %s WHERE run_id = %s",
            (int(rows_fetched), int(rows_validated), status, note, int(run_id)))


# ----------------------------------------------------------------- raw tape

RAW_COLS = ["run_id", "source", "requested_date", "raw_date", "raw_symbol",
            "raw_expiry", "raw_open", "raw_high", "raw_low", "raw_close",
            "raw_volume", "raw_open_interest"]


def write_raw(conn, raw: pd.DataFrame, run_id: int, source: str) -> int:
    """
    COPY the raw tape in, exactly as fetched.

    Append-only and never deduplicated: a duplicate row in the upstream response is
    part of the historical record. Deduplication happens when building `quote`.
    """
    if raw is None or raw.empty:
        return 0
    d = pd.DataFrame({
        "run_id": run_id,
        "source": raw.get("_source", source),
        "requested_date": pd.to_datetime(
            raw.get("_requested_date"), errors="coerce").dt.date
        if "_requested_date" in raw.columns else None,
        "raw_date": raw.get("Date"),
        "raw_symbol": raw.get("Symbol"),
        "raw_expiry": raw.get("ExpiryDate"),
        "raw_open": raw.get("Open"),
        "raw_high": raw.get("High"),
        "raw_low": raw.get("Low"),
        "raw_close": raw.get("Close"),
        "raw_volume": raw.get("Volume"),
        "raw_open_interest": raw.get("OpenInterest"),
    })
    buf = io.StringIO()
    for rec in d.reindex(columns=RAW_COLS).itertuples(index=False):
        out = []
        for v in rec:
            v = _clean(v)
            if v is None:
                out.append(r"\N")
            else:
                s = str(v).replace("\\", "\\\\").replace("\t", " ")
                s = s.replace("\n", " ").replace("\r", " ")
                out.append(s)
        buf.write("\t".join(out) + "\n")
    buf.seek(0)
    stmt = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier("bhavcopy_raw"),
        sql.SQL(", ").join(sql.Identifier(c) for c in RAW_COLS))
    with conn.cursor() as cur, cur.copy(stmt) as cp:
        cp.write(buf.read())
    return len(d)


def read_raw(conn, start: date | None = None, end: date | None = None,
             symbols=None) -> pd.DataFrame:
    """
    Read the raw tape back in the shape a BhavcopySource would return.

    Returns MCX's own column names and formats, so replaying from the database
    exercises the identical parsing and validation path as a live fetch.
    """
    q = ["SELECT requested_date, raw_date AS \"Date\", raw_symbol AS \"Symbol\", "
         "raw_expiry AS \"ExpiryDate\", raw_open AS \"Open\", raw_high AS \"High\", "
         "raw_low AS \"Low\", raw_close AS \"Close\", raw_volume AS \"Volume\", "
         "raw_open_interest AS \"OpenInterest\", source FROM bhavcopy_raw WHERE 1=1"]
    p = []
    if start:
        q.append("AND (requested_date IS NULL OR requested_date >= %s)")
        p.append(start)
    if end:
        q.append("AND (requested_date IS NULL OR requested_date <= %s)")
        p.append(end)
    if symbols:
        q.append("AND btrim(upper(raw_symbol)) = ANY(%s)")
        p.append([s.strip().upper() for s in symbols])
    q.append("ORDER BY raw_id")
    df = pd.read_sql(" ".join(q), conn, params=p) if _has_pandas_sql() else \
        _fetch_df(conn, " ".join(q), p)
    if df.empty:
        return df
    df["_requested_date"] = df.pop("requested_date")
    df["_source"] = df.pop("source")
    return df


def _has_pandas_sql() -> bool:
    return False   # psycopg3 is not a DBAPI pandas special-cases; use _fetch_df


def _fetch_df(conn, query: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query, params or [])
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# -------------------------------------------------------- dimension and fact

def upsert_contracts(conn, tidy: pd.DataFrame) -> int:
    from . import specs
    from .calendar import expiry_window_ok

    g = (tidy.groupby(["contract_id", "symbol", "expiry_date"])
         .agg(first_seen=("date", "min"), last_seen=("date", "max"))
         .reset_index())
    # Resolve each contract's spec as of the date it was first seen. A revision
    # applies to contracts listed after it, so the listing date -- not today -- is
    # what governs, and capturing it here is what freezes history against a later
    # change to the registry.
    sp = [specs.get(sym, on=fs) for sym, fs in zip(g["symbol"], g["first_seen"])]
    g["trading_unit_g"] = [s.trading_unit_g for s in sp]
    g["quote_base_g"] = [s.quote_base_g for s in sp]
    g["purity"] = [s.purity for s in sp]
    g["fine_g_per_lot"] = [s.fine_g_per_lot for s in sp]
    g["tick_size_inr"] = [s.tick_size_inr for s in sp]
    g["deliverable"] = [s.deliverable for s in sp]
    g["expiry_window_ok"] = [expiry_window_ok(s, e)
                             for s, e in zip(g["symbol"], g["expiry_date"])]
    cols = ["contract_id", "symbol", "expiry_date", "trading_unit_g", "quote_base_g",
            "purity", "fine_g_per_lot", "tick_size_inr", "deliverable", "first_seen",
            "last_seen", "expiry_window_ok"]
    return _upsert(conn, "contract", cols, ["contract_id"], g)


QUOTE_COLS = ["contract_id", "trade_date", "symbol", "expiry_date", "open", "high",
              "low", "close", "volume", "open_interest", "norm_close",
              "days_to_expiry", "sessions_to_expiry", "sessions_estimated",
              "t_years", "in_tender", "date_flag", "date_echo", "expiry_window_bad",
              "ohlc_bad", "no_trade", "settle_untradeable", "spec_mismatch",
              "run_id"]


def upsert_quotes(conn, tidy: pd.DataFrame, run_id: int) -> int:
    d = tidy.rename(columns={"date": "trade_date", "T_years": "t_years"}).copy()
    d["run_id"] = run_id
    for c in ("sessions_estimated", "date_echo", "expiry_window_bad", "ohlc_bad",
              "no_trade", "settle_untradeable", "in_tender", "spec_mismatch"):
        if c not in d.columns:
            d[c] = False
    if "norm_close" not in d.columns:
        d["norm_close"] = np.nan
    return _upsert(conn, "quote", QUOTE_COLS, ["contract_id", "trade_date"], d)


def write_dq(conn, run_id: int, report) -> int:
    if not getattr(report, "counts", None):
        return 0
    rows = [{"run_id": run_id, "issue": k, "rows": int(v),
             "detail": str(report.details.get(k + "_examples",
                                              report.details.get(k, "")))[:500] or None}
            for k, v in report.counts.items()]
    return _upsert(conn, "dq_flag", ["run_id", "issue", "rows", "detail"],
                   ["run_id", "issue"], pd.DataFrame(rows))


def read_quotes(conn, start=None, end=None, symbols=None,
                with_specs: bool = True) -> pd.DataFrame:
    """
    Validated quotes back as the tidy frame the model layer expects.

    With `with_specs` (the default) the per-contract specification captured at ingest
    is joined on as `spec_*` columns, and `normalize.add_normalized` prefers it over
    the in-code registry. That is what makes the stored snapshot load-bearing rather
    than decorative: a later specification revision cannot retroactively rewrite the
    normalised price of a contract that was listed under the old rules.
    """
    spec_sel = (", c.trading_unit_g AS spec_trading_unit_g, "
                "c.quote_base_g AS spec_quote_base_g, c.purity AS spec_purity, "
                "c.fine_g_per_lot AS spec_fine_g_per_lot, "
                "c.tick_size_inr AS spec_tick_size_inr, c.first_seen "
                "AS contract_first_seen" if with_specs else "")
    spec_join = ("LEFT JOIN contract c ON c.contract_id = q.contract_id"
                 if with_specs else "")
    q = ["SELECT q.*{} FROM quote q {} WHERE 1=1".format(spec_sel, spec_join)]
    p = []
    # Qualify every column: `contract` also carries symbol and expiry_date, so an
    # unqualified reference becomes ambiguous once the join is present.
    if start:
        q.append("AND q.trade_date >= %s"); p.append(start)
    if end:
        q.append("AND q.trade_date <= %s"); p.append(end)
    if symbols:
        q.append("AND q.symbol = ANY(%s)")
        p.append([s.strip().upper() for s in symbols])
    q.append("ORDER BY q.trade_date, q.symbol, q.expiry_date")
    df = _fetch_df(conn, " ".join(q), p)
    if df.empty:
        return df
    df = df.rename(columns={"trade_date": "date", "t_years": "T_years"})
    numeric = ["open", "high", "low", "close", "norm_close", "T_years",
               "spec_trading_unit_g", "spec_quote_base_g", "spec_purity",
               "spec_fine_g_per_lot", "spec_tick_size_inr"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("volume", "open_interest"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return df


# ------------------------------------------------------------ model outputs

def write_model(conn, daily=None, prem=None, rows=None, sym_resid=None,
                pairs=None) -> dict:
    """Persist one model pass. Every table upserts, so a re-run converges."""
    n = {}
    if daily is not None and not daily.empty:
        d = daily.rename(columns={"date": "trade_date"})
        n["carry_daily"] = _upsert(
            conn, "carry_daily",
            ["trade_date", "level", "slope", "fit_rmse", "n_obs", "n_window_days",
             "has_premium_est", "d_level", "d_slope"], ["trade_date"], d)
    if prem is not None and not prem.empty:
        d = prem.rename(columns={"date": "trade_date"})
        n["symbol_premium"] = _upsert(
            conn, "symbol_premium", ["trade_date", "symbol", "premium_bp"],
            ["trade_date", "symbol"], d)
    if rows is not None and not rows.empty:
        d = rows.rename(columns={"date": "trade_date"})
        cols = ["contract_id", "trade_date", "residual_bp", "fitted_log",
                "fit_weight", "in_fit", "premium_bp", "noise_bp", "half_spread_bp",
                "max_lots", "tradable"]
        for c in cols:
            if c not in d.columns:
                d[c] = None
        n["contract_residual"] = _upsert(
            conn, "contract_residual", cols, ["contract_id", "trade_date"], d)
    if sym_resid is not None and not sym_resid.empty:
        d = sym_resid.rename(columns={"date": "trade_date"})
        n["symbol_residual"] = _upsert(
            conn, "symbol_residual", ["trade_date", "symbol", "resid_bp", "n", "wsum"],
            ["trade_date", "symbol"], d)
    if pairs is not None and not pairs.empty:
        d = pairs.rename(columns={"date": "trade_date"})
        cols = ["trade_date", "pair", "sym_a", "sym_b", "maturity_matched",
                "spread_bp", "z", "dislocation_bp", "centre", "scale", "pctile",
                "halflife", "adf_t", "trail_n", "gate_history", "gate_stationary",
                "gate_halflife", "gate_magnitude", "signal_ok", "direction"]
        for c in cols:
            if c not in d.columns:
                d[c] = None
        # Infinity is a legitimate half-life ("does not revert") but not a valid
        # double in Postgres numeric contexts; store it as NULL and let the gate
        # column carry the meaning.
        d = d.copy()
        for c in ("halflife", "adf_t", "z", "scale"):
            d[c] = pd.to_numeric(d[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
        n["pair_signal"] = _upsert(conn, "pair_signal", cols,
                                   ["trade_date", "pair"], d)
    return n


def write_anomalies(conn, det: pd.DataFrame) -> int:
    """
    Persist the anomaly review queue.

    Feature values are stored as jsonb rather than as columns so the feature set can
    change without a migration -- this is a diagnostic whose inputs are expected to be
    revised, not a stable fact table.
    """
    if det is None or det.empty:
        return 0
    import json

    from . import anomaly as anom
    feats = [c for c in anom.FEATURES if c in det.columns]
    d = det.rename(columns={"date": "trade_date"}).copy()
    d["features"] = [
        json.dumps({k: (None if not np.isfinite(v) else round(float(v), 6))
                    for k, v in rec.items()})
        for rec in d[feats].to_dict(orient="records")
    ]
    cols = ["contract_id", "trade_date", "symbol", "score", "is_anomaly",
            "top_feature", "top_z", "rule_flagged", "features"]
    return _upsert(conn, "dq_anomaly", cols, ["contract_id", "trade_date"], d)


LEDGER_COLS = [
    "model_run_id", "trade_key", "pair", "entry_date", "exit_date", "sessions_held",
    "notional_exposure", "initial_margin", "initial_margin_peak", "span_margin",
    "span_margin_peak", "maintenance_margin", "elm", "additional_margin",
    "delivery_margin", "concentration_margin", "inter_commodity_credit",
    "floor_binds", "total_capital_blocked", "margin_pct_notional",
    "peak_margin_utilisation", "mtm_final", "mtm_worst", "mtm_best",
    "mtm_daily_vol", "min_cash_buffer_required", "margin_call_sessions",
    "was_forced_out", "net_pnl", "return_on_capital_blocked", "exit_reason",
]


def write_position_ledger(conn, summary: pd.DataFrame,
                          model_run_id: int = 0) -> int:
    """
    Persist the per-pair capital ledger.

    Keyed by (model_run_id, trade_key) so successive backtests can be compared: the
    same signal on the same data produces different outcomes at different capital
    levels, and that difference is the point of storing it.
    """
    if summary is None or summary.empty:
        return 0
    d = summary.copy()
    d["model_run_id"] = model_run_id
    # trade_id is an in-process object id; make a stable key for the store
    d["trade_key"] = [
        "{}|{}|{}".format(r.pair, r.entry_date, r.exit_date)
        for r in d.itertuples(index=False)]
    for c in LEDGER_COLS:
        if c not in d.columns:
            d[c] = None
    return _upsert(conn, "bt_position_ledger", LEDGER_COLS,
                   ["model_run_id", "trade_key"], d)


def write_risk_params(conn, files) -> dict:
    """
    Persist published MCX risk parameters, keyed by business date.

    Upserts on (business_date, commodity) and (business_date, leg pair), so reloading
    a file converges rather than accumulating -- and a corrected republication of the
    same day's file replaces the superseded values instead of sitting beside them.
    """
    import pandas as _pd
    if not files:
        return {}
    if not isinstance(files, (list, tuple)):
        files = [files]

    meta, params, credits = [], [], []
    for f in files:
        meta.append({
            "business_date": f.business_date, "kind": f.kind,
            "source_path": f.source_path, "exchange": f.exchange,
            "commodities": len(f.commodities), "credits": len(f.credits),
            "unrecognised": sum(f.unrecognised.values()) if f.unrecognised else 0,
            "notes": "; ".join(f.notes) if f.notes else None})
        for code, c in f.commodities.items():
            try:
                scan = c.scan_pct()
            except Exception:
                scan = None
            params.append({
                "business_date": f.business_date, "commodity": code,
                "scan_pct": scan, "scan_basis": c.scan_basis,
                "intra_spread_charge": c.intra_spread_charge,
                "spot_month_charge": c.spot_month_charge,
                "elm_pct": c.elm_pct, "additional_pct": c.additional_pct,
                "delivery_pct": c.delivery_pct, "source_path": f.source_path})
        for cr in f.credits:
            a, b = sorted((cr.leg_a, cr.leg_b))
            credits.append({
                "business_date": f.business_date, "leg_a": a, "leg_b": b,
                "credit_rate": cr.credit_rate, "ratio_a": cr.ratio_a,
                "ratio_b": cr.ratio_b, "priority": cr.priority,
                "source_path": f.source_path})

    out = {}
    if meta:
        out["risk_param_file"] = _upsert(
            conn, "risk_param_file",
            ["business_date", "kind", "source_path", "exchange", "commodities",
             "credits", "unrecognised", "notes"],
            ["business_date", "kind", "source_path"], _pd.DataFrame(meta))
    if params:
        out["risk_param"] = _upsert(
            conn, "risk_param",
            ["business_date", "commodity", "scan_pct", "scan_basis",
             "intra_spread_charge", "spot_month_charge", "elm_pct",
             "additional_pct", "delivery_pct", "source_path"],
            ["business_date", "commodity"], _pd.DataFrame(params))
    if credits:
        out["risk_spread_credit"] = _upsert(
            conn, "risk_spread_credit",
            ["business_date", "leg_a", "leg_b", "credit_rate", "ratio_a", "ratio_b",
             "priority", "source_path"],
            ["business_date", "leg_a", "leg_b"], _pd.DataFrame(credits))
    return out


def read_risk_params(conn, start=None, end=None) -> pd.DataFrame:
    q = "SELECT * FROM risk_param"
    args = []
    if start is not None and end is not None:
        q += " WHERE business_date BETWEEN %s AND %s"
        args = [start, end]
    return _fetch_df(conn, q + " ORDER BY business_date, commodity", args or None)


def read_spread_credits(conn, asof=None) -> pd.DataFrame:
    """
    Published cross-margin credits in force on a date.

    "In force" means the newest file dated ON OR BEFORE the session -- never a later
    one. A credit published after the session did not exist when the position was
    margined, and using it would be look-ahead.
    """
    if asof is None:
        return _fetch_df(conn, "SELECT * FROM risk_spread_credit "
                               "ORDER BY business_date, leg_a, leg_b")
    return _fetch_df(
        conn,
        "SELECT DISTINCT ON (leg_a, leg_b) * FROM risk_spread_credit "
        "WHERE business_date <= %s "
        "ORDER BY leg_a, leg_b, business_date DESC", [asof])


def published_scan_schedule(conn) -> dict:
    """{(business_date, commodity): scan_pct} for the backtest to walk forward on."""
    df = read_risk_params(conn)
    if df.empty:
        return {}
    df = df[df["scan_pct"].notna()]
    return {(r.business_date, r.commodity): float(r.scan_pct)
            for r in df.itertuples(index=False)}


RISK_BOARD_COLS = [
    "model_run_id", "asof", "pair", "account_capital", "lots_a", "lots_b",
    "direction", "notional_exposure", "span_margin", "initial_margin", "elm",
    "delivery_margin", "concentration_margin", "inter_commodity_credit",
    "floor_binds", "marginal_margin", "book_required", "required_if_added",
    "free_cash_if_added", "margin_utilisation", "utilisation_if_added",
    "buffer_mult", "buffer_state", "span_change_pct", "im_change_pct",
    "scan_range_pct", "scan_range_prior_pct", "lookback_from", "z", "spread_bp",
    "signal_ok", "sessions_to_expiry", "action", "reason",
]


def write_risk_board(conn, board: pd.DataFrame, model_run_id: int = 0,
                     account_capital: float | None = None) -> int:
    """
    Persist the live risk board.

    Keyed by (model_run_id, asof, pair) so the same session can be re-evaluated at a
    different capital level and both records kept. The instruction is a decision, and
    a decision stored without the numbers behind it cannot be reviewed later.
    """
    if board is None or board.empty:
        return 0
    d = board.copy()
    d["model_run_id"] = model_run_id
    if account_capital is not None:
        d["account_capital"] = float(account_capital)
    d = d.rename(columns={"date": "asof"})
    for c in RISK_BOARD_COLS:
        if c not in d.columns:
            d[c] = None
    return _upsert(conn, "risk_board", RISK_BOARD_COLS,
                   ["model_run_id", "asof", "pair"], d)


def risk_board(conn, model_run_id: int | None = None) -> pd.DataFrame:
    if model_run_id is None:
        return _fetch_df(conn, "SELECT * FROM risk_board "
                               "WHERE model_run_id = (SELECT max(model_run_id) "
                               "FROM risk_board) ORDER BY asof, pair")
    return _fetch_df(conn, "SELECT * FROM risk_board WHERE model_run_id = %s "
                           "ORDER BY asof, pair", [int(model_run_id)])


def position_ledger(conn, model_run_id: int | None = None) -> pd.DataFrame:
    if model_run_id is None:
        return _fetch_df(conn, "SELECT * FROM bt_position_ledger "
                               "WHERE model_run_id = (SELECT max(model_run_id) "
                               "FROM bt_position_ledger) ORDER BY entry_date")
    return _fetch_df(conn, "SELECT * FROM bt_position_ledger WHERE model_run_id = %s "
                           "ORDER BY entry_date", [int(model_run_id)])


def anomaly_review(conn, limit: int = 50) -> pd.DataFrame:
    """Worst unexplained anomalies first -- the rows a human should actually look at."""
    return _fetch_df(conn, "SELECT * FROM v_anomaly_review LIMIT %s", [int(limit)])


def write_backtest(conn, trades=None, bt_daily=None, model_run_id: int = 0) -> dict:
    n = {}
    if trades is not None and not trades.empty:
        d = trades.copy()
        d["model_run_id"] = model_run_id
        cols = ["model_run_id", "pair", "sym_a", "sym_b", "cid_a", "cid_b",
                "dir_pair", "lots_a", "lots_b", "entry_date", "exit_date",
                "sessions_held", "entry_z", "exit_z", "entry_spread_bp",
                "exit_spread_bp", "notional_a", "notional_b", "fine_residual_g",
                "residual_frac", "gross_pnl", "total_cost", "net_pnl", "gross_bp",
                "net_bp", "exit_reason", "illiquid_exit", "entry_margin",
                "return_on_margin"]
        for c in cols:
            if c not in d.columns:
                d[c] = None
        d = d.copy()
        for c in ("entry_z", "exit_z"):
            d[c] = pd.to_numeric(d[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
        data = _rows(d, cols)
        stmt = sql.SQL("INSERT INTO bt_trade ({}) VALUES ({})").format(
            sql.SQL(", ").join(sql.Identifier(c) for c in cols),
            sql.SQL(", ").join(sql.Placeholder() * len(cols)))
        with conn.cursor() as cur:
            cur.executemany(stmt, data)
        n["bt_trade"] = len(data)
    if bt_daily is not None and not bt_daily.empty:
        d = bt_daily.rename(columns={"date": "trade_date"}).copy()
        d["model_run_id"] = model_run_id
        n["bt_daily"] = _upsert(
            conn, "bt_daily",
            ["model_run_id", "trade_date", "realised_cum", "unrealised", "equity",
             "day_realised", "day_cost", "n_open", "ret", "margin", "span",
             "inter_credit"],
            ["model_run_id", "trade_date"], d)
    return n


# ---------------------------------------------------------------- convenience

def snapshot(conn) -> pd.DataFrame:
    return _fetch_df(conn, "SELECT * FROM v_snapshot")


def signal_board(conn, on: date | None = None) -> pd.DataFrame:
    if on is None:
        return _fetch_df(conn, "SELECT * FROM v_signal_board WHERE trade_date = "
                               "(SELECT max(trade_date) FROM pair_signal)")
    return _fetch_df(conn, "SELECT * FROM v_signal_board WHERE trade_date = %s", [on])


def echo_audit(conn) -> pd.DataFrame:
    return _fetch_df(conn, "SELECT * FROM v_echo_audit")


def table_counts(conn) -> pd.DataFrame:
    tables = ["ingest_run", "bhavcopy_raw", "contract", "quote", "dq_flag",
              "dq_anomaly", "carry_daily", "symbol_premium", "contract_residual",
              "symbol_residual", "pair_signal", "bt_trade", "bt_daily",
              "bt_position_ledger", "risk_board", "risk_param_file",
              "risk_param", "risk_spread_credit"]
    out = []
    with conn.cursor() as cur:
        for t in tables:
            try:
                cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(t)))
                out.append({"table": t, "rows": cur.fetchone()[0]})
            except psycopg.Error:
                conn.rollback()
                out.append({"table": t, "rows": None})
    return pd.DataFrame(out)
