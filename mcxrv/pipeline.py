"""
The analysis pipeline as a callable, for code that needs the results rather than the
printed report: the nightly job, the API snapshot, tests.

Same steps and the same configuration defaults as scripts/run_analysis.py
(ingest -> normalise -> carry -> liquidity -> signal -> backtest -> risk board), so a
number on the dashboard and a number in the written report come from one code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from mcxrv import backtest, carry, costs, ingest, liquidity, margin, normalize, risk
from mcxrv import signal as sig_mod
from mcxrv import specs


@dataclass
class PipelineConfig:
    source: str = "synthetic"          # synthetic | csv | live | db
    start: date = date(2024, 1, 1)
    end: date = date(2026, 9, 18)
    csv_dir: str = "data/raw"
    seed: int = 7
    null_world: bool = False           # synthetic only: residual_bp = 0
    z_entry: float = 2.0
    half_spread_bp: float | None = None
    maturity_matched_only: bool = False
    account_capital: float = 1_000_000.0
    margin_limit: float | None = None
    span_params_file: str | None = None


@dataclass
class PipelineResult:
    config: PipelineConfig
    report: ingest.DataQualityReport
    quotes: pd.DataFrame               # validated, normalised contract-day rows
    rows: pd.DataFrame                 # carry-fitted + liquidity-annotated rows
    carry_daily: pd.DataFrame
    premiums: pd.DataFrame
    pairs: pd.DataFrame                # pair signals, every session
    trades: pd.DataFrame
    bt_daily: pd.DataFrame
    performance: dict
    board: pd.DataFrame                # risk board for the latest session
    span_source: str = "assumed"
    notes: dict = field(default_factory=dict)

    @property
    def asof(self):
        if self.pairs is not None and not self.pairs.empty:
            return max(self.pairs["date"])
        if self.quotes is not None and not self.quotes.empty:
            return max(self.quotes["date"])
        return None


def load_quotes(cfg: PipelineConfig):
    """Raw -> validated quotes. Returns (quotes, DataQualityReport)."""
    if cfg.source == "db":
        from mcxrv.db import repo as dbrepo
        with dbrepo.connect() as c:
            df = dbrepo.read_quotes(c, start=cfg.start, end=cfg.end,
                                    symbols=specs.ALL_SYMBOLS)
        rep = ingest.DataQualityReport(rows_in=len(df), rows_out=len(df))
        rep.note("source", "postgres (already validated on ingest)")
        return df, rep

    if cfg.source == "synthetic":
        from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource
        scfg = SyntheticConfig(start=cfg.start, end=cfg.end, seed=cfg.seed,
                               residual_bp=0.0 if cfg.null_world else 22.0)
        raw = SyntheticSource(scfg).fetch_range(cfg.start, cfg.end)
    elif cfg.source == "csv":
        from mcxrv.sources.csv_dir import CSVDirectorySource
        raw = CSVDirectorySource(cfg.csv_dir).load_all(cfg.start, cfg.end)
    elif cfg.source == "live":
        from mcxrv.sources.mcx_live import MCXLiveSource
        raw = MCXLiveSource().fetch_range(cfg.start, cfg.end, symbols=specs.ALL_SYMBOLS)
    else:
        raise ValueError("unknown source: {}".format(cfg.source))

    df, rep, _ = ingest.build_dataset(raw, symbols=specs.ALL_SYMBOLS)
    return df, rep


def run(cfg: PipelineConfig | None = None, quotes: pd.DataFrame | None = None,
        report: ingest.DataQualityReport | None = None) -> PipelineResult:
    """Run the full pipeline. Pass `quotes` to skip loading (tests, replays)."""
    cfg = cfg or PipelineConfig()
    if quotes is None:
        quotes, report = load_quotes(cfg)
    report = report or ingest.DataQualityReport(rows_in=len(quotes), rows_out=len(quotes))
    empty = pd.DataFrame()
    if quotes is None or quotes.empty:
        return PipelineResult(cfg, report, empty, empty, empty, empty, empty, empty,
                              empty, {}, empty, notes={"stopped": "no usable rows"})

    quotes = normalize.add_normalized(quotes)
    rows, daily, prem = carry.fit(quotes, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())

    sym_resid = carry.symbol_residual(rows)
    scfg = sig_mod.SignalConfig(z_entry=cfg.z_entry)
    pairs = sig_mod.pair_spreads(sym_resid, scfg,
                                 maturity_matched_only=cfg.maturity_matched_only)

    ccfg = costs.CostConfig(half_spread_bp_override=cfg.half_spread_bp)
    sp = (margin.SpanParameters.from_mcx_file(cfg.span_params_file)
          if cfg.span_params_file else
          margin.SpanParameters(price_scan_range_pct=margin.derive_scan_range(rows)))
    msched = margin.volatility_schedule(rows)
    bcfg = backtest.BacktestConfig(exit_z=scfg.z_exit, span_params=sp,
                                   margin_schedule=msched,
                                   margin_limit=cfg.margin_limit,
                                   account_capital=cfg.account_capital)

    trades, bt_daily, perf = empty, empty, {}
    board = empty
    if not pairs.empty:
        # Settlement is published after the close: fill at t+1, never at t.
        pairs_exec = sig_mod.lag_for_execution(pairs, sessions=1)
        trades, bt_daily, _diag, _pos = backtest.run(rows, pairs_exec, bcfg, ccfg)
        perf = backtest.performance(trades, bt_daily, bcfg)

        asof = max(pairs["date"])
        book = risk.open_book_from_trades(rows, trades, asof)
        board = risk.board(rows, pairs,
                           # Same runway rule the backtest enters with, so a live
                           # alert never proposes a trade the backtest would refuse.
                           risk.RiskBoardConfig(
                               account_capital=cfg.account_capital,
                               min_sessions_to_expiry=bcfg.min_sessions_to_expiry_entry),
                           span_params=sp, margin_schedule=msched,
                           bt_cfg=bcfg, open_book=book)

    return PipelineResult(cfg, report, quotes, rows, daily, prem, pairs, trades,
                          bt_daily, perf, board, span_source=str(sp.source),
                          notes={"z_entry": cfg.z_entry, "z_exit": scfg.z_exit})
