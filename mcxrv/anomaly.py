"""
Unsupervised data-quality anomaly detection with an Isolation Forest.

WHY THIS IS SEPARATE FROM THE SIGNAL, AND MUST STAY SEPARATE
------------------------------------------------------------
This module produces a REVIEW QUEUE, not a trading input. `ingest` catches the
pathologies that were anticipated well enough to write a rule for: a date echo, a
swapped date field, a padded symbol, a zero-volume derived settlement. It cannot
catch what nobody anticipated. An unsupervised detector can, because it only needs
"unlike the rest of the data" rather than a specification.

The flags here are deliberately NON-BLOCKING and are never consulted by `carry`,
`signal` or `backtest`. That restraint is not fussiness -- it is the difference
between a sound backtest and a broken one:

  * The forest is fit on the WHOLE sample, because a data-quality audit is a
    retrospective question about the dataset, not a forecast. That is legitimate for
    flagging. It would be flat look-ahead if the flags then filtered a backtest,
    because a trade on day t would be excluded using information from later days.
  * Dropping the observations a model finds surprising also quietly deletes real
    market events -- a genuine gap, a genuine liquidity shock -- and those are
    exactly the days a relative-value strategy lives or dies on. Removing them
    flatters the result.

So: score everything, flag generously, block nothing. A human decides what an
individual flag means, and the rule-based validators in `ingest` remain the only
thing that actually gates data.

Features are chosen to describe an observation's relationship to its own history and
to its peers on the same session, since a corrupt print is usually wrong relative to
one or the other rather than implausible in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURES = [
    "log_ret",          # within-contract close-to-close move
    "ret_vs_peers_bp",  # own move minus the median move of all other contracts
    "peer_dev_bp",      # normalised price vs the cross-sectional median that session
    "range_frac",       # (high - low) / close
    "close_in_range",   # where the close sits inside the day's range
    "d_log_volume",     # change in traded volume
    "d_log_oi",         # change in open interest
    "ret_in_ticks",     # |move| measured in ticks: catches implausible jumps
    "resid_bp",         # carry-model residual, where one exists
]


@dataclass
class AnomalyConfig:
    contamination: float = 0.01   # expected share of anomalies; a prior, not a fact
    n_estimators: int = 300
    max_samples: str | int = "auto"
    random_state: int = 0
    min_rows: int = 200
    # Exclude rows already flagged by a rule. They are known-bad, so leaving them in
    # would train the forest to find what `ingest` has already found, wasting its
    # capacity on the problems that are solved.
    exclude_rule_flagged: bool = True
    features: list = field(default_factory=lambda: list(FEATURES))
    # Worker processes for the forest fit. -1 uses all cores, which is measurably
    # faster here (1.1 s against 2.1 s on 27k rows by 300 trees) and is the default.
    # Exposed because joblib spawns real processes, so an environment that restricts
    # them needs a way to ask for a single-threaded fit -- not because anything is
    # known to be wrong with the parallel path.
    n_jobs: int = -1


def available() -> bool:
    """scikit-learn is optional; callers degrade gracefully without it."""
    try:
        from sklearn.ensemble import IsolationForest  # noqa: F401
        return True
    except ImportError:
        return False


def build_features(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Per (contract, session) feature matrix.

    Two of these carry most of the weight. `peer_dev_bp` compares a contract's
    normalised price with the cross-sectional median of everything else trading that
    session -- a single mispriced contract is wrong relative to its peers even when
    its own time series looks smooth. `ret_in_ticks` expresses the move in tick units,
    which is what separates a real 40 bp move in GOLDPETAL (a handful of ticks) from a
    decimal-point error (thousands).
    """
    d = rows.sort_values(["contract_id", "date"]).copy()

    g = d.groupby("contract_id", sort=False)
    ln = np.log(d["norm_close"].astype(float))
    d["log_ret"] = ln.groupby(d["contract_id"], sort=False).diff()

    d["d_log_volume"] = (np.log1p(d["volume"].astype(float))
                         .groupby(d["contract_id"], sort=False).diff())
    d["d_log_oi"] = (np.log1p(d["open_interest"].astype(float))
                     .groupby(d["contract_id"], sort=False).diff())

    hi = d["high"].astype(float)
    lo = d["low"].astype(float)
    cl = d["close"].astype(float)
    rng = (hi - lo)
    d["range_frac"] = (rng / cl.replace(0, np.nan)).fillna(0.0)
    d["close_in_range"] = np.where(rng.abs() > 1e-9, (cl - lo) / rng, 0.5)

    # cross-sectional comparisons, per session
    med_ln = ln.groupby(d["date"], sort=False).transform("median")
    d["peer_dev_bp"] = (ln - med_ln) * 10000.0
    med_ret = d["log_ret"].groupby(d["date"], sort=False).transform("median")
    d["ret_vs_peers_bp"] = (d["log_ret"] - med_ret) * 10000.0

    tick_bp = d.get("tick_bp_of_norm")
    if tick_bp is None:
        tick_bp = pd.Series(1.0, index=d.index)
    tick_bp = pd.to_numeric(tick_bp, errors="coerce").replace(0, np.nan)
    d["ret_in_ticks"] = (d["log_ret"].abs() * 10000.0 / tick_bp).fillna(0.0)

    if "residual_bp" not in d.columns:
        d["residual_bp"] = np.nan
    d["resid_bp"] = pd.to_numeric(d["residual_bp"], errors="coerce")

    return d


def _rule_flagged(d: pd.DataFrame) -> pd.Series:
    cols = [c for c in ("date_echo", "expiry_window_bad", "ohlc_bad", "no_trade",
                        "settle_untradeable") if c in d.columns]
    if not cols:
        return pd.Series(False, index=d.index)
    return d[cols].fillna(False).any(axis=1)


def detect(rows: pd.DataFrame, cfg: AnomalyConfig | None = None) -> pd.DataFrame:
    """
    Score every observation and flag the most anomalous.

    Returns one row per (contract_id, date) with:
      score        Isolation Forest decision_function. LOWER is more anomalous;
                   the sign convention is sklearn's, kept as-is so the number means
                   what its documentation says.
      is_anomaly   the forest's own call at the configured contamination
      top_feature  the feature furthest from its median in robust (MAD) units, which
                   is what makes a flag actionable -- "anomalous" alone tells a human
                   nothing about where to look
      top_z        that feature's robust deviation
      rule_flagged whether `ingest` had already caught this row by rule

    Returns an empty frame when scikit-learn is absent or there is too little data,
    rather than raising: this is an optional diagnostic, not a pipeline dependency.
    """
    cfg = cfg or AnomalyConfig()
    if not available() or rows is None or rows.empty:
        return pd.DataFrame()

    d = build_features(rows)
    feats = [f for f in cfg.features if f in d.columns]
    X = d[feats].copy()

    # A missing feature means "no prior observation for this contract", not an
    # outlier. Impute at the column median so a first-of-series row is not flagged
    # merely for being first.
    med = X.median(numeric_only=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)

    fit_mask = pd.Series(True, index=d.index)
    if cfg.exclude_rule_flagged:
        fit_mask &= ~_rule_flagged(d)
    if int(fit_mask.sum()) < cfg.min_rows:
        fit_mask = pd.Series(True, index=d.index)
    if len(X) < cfg.min_rows:
        return pd.DataFrame()

    from sklearn.ensemble import IsolationForest
    forest = IsolationForest(
        n_estimators=cfg.n_estimators, contamination=cfg.contamination,
        max_samples=cfg.max_samples, random_state=cfg.random_state,
        n_jobs=cfg.n_jobs)
    forest.fit(X[fit_mask])

    score = forest.decision_function(X)
    flag = forest.predict(X) == -1

    # Robust z per feature, for attribution. MAD rather than sd so the scale is not
    # itself set by the outliers we are trying to describe.
    mad = (X - X.median()).abs().median().replace(0, np.nan)
    z = ((X - X.median()).abs() / (1.4826 * mad)).fillna(0.0)
    top_feature = z.idxmax(axis=1)
    top_z = z.max(axis=1)

    out = pd.DataFrame({
        "contract_id": d["contract_id"].to_numpy(),
        "date": d["date"].to_numpy(),
        "symbol": d["symbol"].to_numpy(),
        "score": score,
        "is_anomaly": flag,
        "top_feature": top_feature.to_numpy(),
        "top_z": top_z.to_numpy(),
        "rule_flagged": _rule_flagged(d).to_numpy(),
    })
    for f in feats:
        out[f] = X[f].to_numpy()
    return out.sort_values("score").reset_index(drop=True)


def summary(det: pd.DataFrame) -> pd.DataFrame:
    """
    Per-symbol anomaly rates, split by whether a rule had already caught the row.

    `flagged_new` is the column that justifies this module: rows the forest found
    that no rule anticipated. If it were always zero, the rule set would already be
    sufficient and this could be deleted.
    """
    if det.empty:
        return pd.DataFrame()
    out = []
    for sym, g in det.groupby("symbol"):
        flagged = g[g["is_anomaly"]]
        out.append({
            "symbol": sym,
            "rows": len(g),
            "flagged": int(g["is_anomaly"].sum()),
            "flag_rate": round(float(g["is_anomaly"].mean()), 4),
            "flagged_already_known": int((flagged["rule_flagged"]).sum()),
            "flagged_new": int((~flagged["rule_flagged"]).sum()),
            "worst_score": round(float(g["score"].min()), 4),
        })
    return pd.DataFrame(out).sort_values("symbol").reset_index(drop=True)


def top_reasons(det: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """The most anomalous rows a human has not already been told about."""
    if det.empty:
        return pd.DataFrame()
    new = det[det["is_anomaly"] & ~det["rule_flagged"]]
    cols = ["date", "symbol", "contract_id", "score", "top_feature", "top_z",
            "log_ret", "peer_dev_bp", "ret_in_ticks", "d_log_volume"]
    cols = [c for c in cols if c in new.columns]
    return new.nsmallest(n, "score")[cols].reset_index(drop=True)


def feature_reason_counts(det: pd.DataFrame) -> pd.DataFrame:
    """Which feature drives the flags -- a sanity check on what the forest is doing."""
    if det.empty:
        return pd.DataFrame()
    flagged = det[det["is_anomaly"]]
    if flagged.empty:
        return pd.DataFrame()
    c = (flagged.groupby(["top_feature", "rule_flagged"]).size()
         .unstack(fill_value=0))
    c.columns = ["new" if not k else "already_known" for k in c.columns]
    return c.reset_index().sort_values(c.columns[-1], ascending=False)
