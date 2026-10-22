"""
Turn the carry residual into a gated signal -- and refuse to emit one when the
evidence does not support it.

Everything here is strictly trailing. Every mean, standard deviation, half-life
and stationarity statistic on day t is computed from data up to t-1 only. This is
the leak that sinks most projects of this kind: computing a z-score against the
FULL-SAMPLE mean and standard deviation uses the future to judge the present, and
it is invisible in the output because the resulting series looks entirely normal.

The gates below exist to keep the product quiet. A dashboard that always has an
opinion is evidence against itself.

  1. history       enough trailing observations to estimate a distribution at all
  2. stationarity  the residual must actually mean-revert (ADF on the window).
                   Without this, a z-score on a drifting series fires forever.
  3. half-life     convergence must be able to happen INSIDE the contract's
                   remaining life. A 40-session half-life in a contract with 12
                   sessions left is untradeable however extreme the z-score.
  4. magnitude     |z| above threshold AND the dislocation must exceed the
                   measurement noise floor -- tick quantisation plus fit error.
                   A 5 bp signal on a 20 bp noise floor is not a signal.
  5. liquidity     both legs must have genuinely traded (see `liquidity`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import specs

# MacKinnon critical values, constant-no-trend, large sample.
ADF_CRIT = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


@dataclass
class SignalConfig:
    lookback: int = 250           # trailing window for mean / sd / diagnostics
    min_history: int = 90         # refuse to score before this many observations
    z_entry: float = 2.0
    z_exit: float = 0.5
    max_halflife_sessions: float = 30.0
    require_stationary: bool = True
    adf_alpha: float = 0.10       # lenient: a hard test on 250 points is brittle
    adf_lags: int = 1
    min_abs_bp: float = 15.0      # absolute floor, on top of the z threshold
    noise_multiple: float = 1.0   # dislocation must exceed noise_floor * this
    use_robust_scale: bool = True  # MAD instead of sd: outlier-resistant


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def adf_tstat(y: np.ndarray, lags: int = 1) -> float:
    """
    Augmented Dickey-Fuller t-statistic, constant, no trend.

    Regress dy_t on [1, y_{t-1}, dy_{t-1}, ...]. The t-stat on y_{t-1} is the
    statistic; more negative means stronger mean reversion. Compare to ADF_CRIT --
    these are NOT normal critical values, which is why a hand-rolled t-test
    against 1.96 would badly over-reject.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < lags + 12:
        return np.nan
    dy = np.diff(y)
    ylag = y[:-1]
    rows = n - 1 - lags
    if rows < 8:
        return np.nan
    X = [np.ones(rows), ylag[lags:]]
    for k in range(1, lags + 1):
        X.append(dy[lags - k:-k] if k < lags + 1 else dy[:rows])
    X = np.column_stack([x[:rows] for x in X])
    yv = dy[lags:][:rows]
    try:
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    resid = yv - X @ beta
    dof = rows - X.shape[1]
    if dof <= 0:
        return np.nan
    s2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan
    se = np.sqrt(max(s2 * xtx_inv[1, 1], 1e-30))
    return float(beta[1] / se)


def halflife_sessions(y: np.ndarray) -> float:
    """
    Mean-reversion half-life in sessions, from an AR(1) fit.

    y_t - y_{t-1} = lam * y_{t-1} + c  =>  half-life = -ln(2) / ln(1 + lam).
    Returns inf for a non-reverting or explosive series: the honest answer is
    "this does not converge", not a large finite number.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 20:
        return np.nan
    dy = np.diff(y)
    ylag = y[:-1]
    X = np.column_stack([np.ones(len(ylag)), ylag])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    lam = float(beta[1])
    phi = 1.0 + lam
    if phi <= 0 or phi >= 1:
        return np.inf
    return float(-np.log(2.0) / np.log(phi))


# ---------------------------------------------------------------------------
# trailing scoring
# ---------------------------------------------------------------------------

def _trailing_stats(series: pd.Series, cfg: SignalConfig) -> pd.DataFrame:
    """
    For each t, statistics computed on the window ENDING AT t-1.

    The `.shift(1)` after every rolling call is the whole no-look-ahead argument
    for this module. Without it, today's observation is inside its own reference
    distribution, which shrinks extreme z-scores precisely when they matter.
    """
    s = series.astype(float)
    roll = s.rolling(cfg.lookback, min_periods=cfg.min_history)

    mean = roll.mean().shift(1)
    sd = roll.std(ddof=1).shift(1)
    med = roll.median().shift(1)
    mad = roll.apply(lambda a: np.nanmedian(np.abs(a - np.nanmedian(a))),
                     raw=True).shift(1)
    # 1.4826 * MAD is a consistent estimator of sigma for a normal.
    sd_robust = 1.4826 * mad
    n = roll.count().shift(1)

    centre = med if cfg.use_robust_scale else mean
    scale = sd_robust if cfg.use_robust_scale else sd
    scale = scale.where(scale > 1e-9)

    out = pd.DataFrame({
        "trail_mean": mean, "trail_sd": sd, "trail_median": med,
        "trail_sd_robust": sd_robust, "trail_n": n,
        "centre": centre, "scale": scale,
    })
    out["z"] = (s - centre) / scale
    out["dislocation_bp"] = s - centre
    # percentile of today's value within the trailing window
    out["pctile"] = roll.apply(
        lambda a: float((a[:-1] <= a[-1]).mean()) if len(a) > 1 else np.nan,
        raw=True)
    return out


def score_symbols(sym_resid: pd.DataFrame, cfg: SignalConfig | None = None,
                  noise_floor: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Score each symbol's dislocation against its own trailing history.

    `sym_resid` is the output of `carry.symbol_residual`. `noise_floor` may carry
    a per (date, symbol) `noise_bp` column; where absent, the gate falls back to
    `cfg.min_abs_bp` alone.
    """
    cfg = cfg or SignalConfig()
    parts = []
    for sym, g in sym_resid.sort_values("date").groupby("symbol"):
        g = g.reset_index(drop=True)
        st = _trailing_stats(g["resid_bp"], cfg)
        g = pd.concat([g, st], axis=1)

        # trailing diagnostics, recomputed on a coarse grid then forward-filled:
        # ADF and AR(1) on a 250-point window every single day is wasteful and the
        # statistics barely move day to day.
        hl, adf = np.full(len(g), np.nan), np.full(len(g), np.nan)
        step = 5
        vals = g["resid_bp"].to_numpy()
        for i in range(cfg.min_history, len(g), step):
            w = vals[max(0, i - cfg.lookback):i]      # strictly before i
            hl[i] = halflife_sessions(w)
            adf[i] = adf_tstat(w, cfg.adf_lags)
        g["halflife"] = pd.Series(hl).ffill()
        g["adf_t"] = pd.Series(adf).ffill()
        g["symbol"] = sym
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    if noise_floor is not None and not noise_floor.empty:
        out = out.merge(noise_floor, on=["date", "symbol"], how="left")
    if "noise_bp" not in out.columns:
        out["noise_bp"] = np.nan

    crit = ADF_CRIT.get(cfg.adf_alpha, -2.57)
    out["gate_history"] = out["trail_n"] >= cfg.min_history
    out["gate_stationary"] = (~cfg.require_stationary) | (out["adf_t"] <= crit)
    out["gate_halflife"] = out["halflife"] <= cfg.max_halflife_sessions
    floor = np.fmax(out["noise_bp"].fillna(0.0) * cfg.noise_multiple, cfg.min_abs_bp)
    out["noise_floor_bp"] = floor
    out["gate_magnitude"] = (out["z"].abs() >= cfg.z_entry) & \
                            (out["dislocation_bp"].abs() >= floor)

    out["signal_ok"] = (out["gate_history"] & out["gate_stationary"]
                        & out["gate_halflife"] & out["gate_magnitude"])
    out["direction"] = np.where(out["signal_ok"],
                                np.where(out["z"] > 0, -1, 1), 0)
    # direction = -1 means the symbol is RICH: sell it, buy the reference.
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def pair_spreads(sym_resid: pd.DataFrame, cfg: SignalConfig | None = None,
                 pairs=None, maturity_matched_only: bool = False) -> pd.DataFrame:
    """
    Residual spreads between symbol pairs, scored on trailing history.

    The spread of two residuals is cleaner than either alone, because the common
    level and carry have already been removed from both and the pair differencing
    cancels any residual mis-estimation of the shared factors.

    `maturity_matched_only` restricts to pairs sharing an expiry window -- the
    GOLDTEN / GOLDGUINEA / GOLDPETAL triangle. Pairs involving GOLDM are still
    valid here, because the carry model has already bridged the ~23-day expiry
    gap; the flag exists so a report can show results both ways and let a reader
    see how much rests on the carry bridge.
    """
    cfg = cfg or SignalConfig()
    wide = sym_resid.pivot_table(index="date", columns="symbol", values="resid_bp")
    syms = [s for s in specs.ALL_SYMBOLS if s in wide.columns]

    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(syms) for b in syms[i + 1:]]
    if maturity_matched_only:
        pairs = [(a, b) for a, b in pairs if specs.is_maturity_matched(a, b)]

    parts = []
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        s = (wide[a] - wide[b]).dropna()
        if len(s) < cfg.min_history + 10:
            continue
        g = pd.DataFrame({"date": s.index, "spread_bp": s.to_numpy()})
        st = _trailing_stats(g["spread_bp"], cfg)
        g = pd.concat([g, st], axis=1)

        hl, adf = np.full(len(g), np.nan), np.full(len(g), np.nan)
        vals = g["spread_bp"].to_numpy()
        for i in range(cfg.min_history, len(g), 5):
            w = vals[max(0, i - cfg.lookback):i]
            hl[i] = halflife_sessions(w)
            adf[i] = adf_tstat(w, cfg.adf_lags)
        g["halflife"] = pd.Series(hl).ffill()
        g["adf_t"] = pd.Series(adf).ffill()

        g["sym_a"], g["sym_b"] = a, b
        g["pair"] = "{}/{}".format(a, b)
        g["maturity_matched"] = specs.is_maturity_matched(a, b)
        parts.append(g)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    crit = ADF_CRIT.get(cfg.adf_alpha, -2.57)
    out["gate_history"] = out["trail_n"] >= cfg.min_history
    out["gate_stationary"] = (~cfg.require_stationary) | (out["adf_t"] <= crit)
    out["gate_halflife"] = out["halflife"] <= cfg.max_halflife_sessions
    out["gate_magnitude"] = (out["z"].abs() >= cfg.z_entry) & \
                            (out["dislocation_bp"].abs() >= cfg.min_abs_bp)
    out["signal_ok"] = (out["gate_history"] & out["gate_stationary"]
                        & out["gate_halflife"] & out["gate_magnitude"])
    # direction = +1: spread is too LOW -> buy a, sell b. -1: the reverse.
    out["direction"] = np.where(out["signal_ok"],
                                np.where(out["z"] > 0, -1, 1), 0)
    return out.sort_values(["date", "pair"]).reset_index(drop=True)


def lag_for_execution(sig: pd.DataFrame, sessions: int = 1,
                      date_col: str = "date", group_col: str = "pair") -> pd.DataFrame:
    """
    Shift signals forward so they are acted on AFTER the data that produced them.

    A settlement price is published after the close. A signal computed from day t's
    settlement therefore cannot be filled at day t's settlement -- that is
    same-bar execution, and it is look-ahead even though every input was trailing.
    The earliest honest fill is the next session.

    This also removes a subtler contamination that matters even more for inference.
    Signed convergence measured as -sign(z[t]) * (spread[t+h] - spread[t]) contains
    the term +|spread[t] - centre|, which is positive by construction. Any noise at
    all in spread[t] therefore produces apparent mean reversion, because
    conditioning on a large deviation guarantees the next draw is closer to the
    mean. Measuring from t+1 instead breaks the shared noise term between the
    signal and the entry price, and the effect disappears.

    Diagnostic signature, worth remembering: noise-driven pseudo-convergence is
    FLAT in horizon and fully realised at h=1. A genuine mean-reverting signal
    GROWS with horizon toward its half-life. Comparing the two profiles separates
    them more reliably than any single t-statistic.
    """
    if sig.empty or sessions <= 0:
        return sig
    out = []
    for _, g in sig.sort_values(date_col).groupby(group_col):
        g = g.sort_values(date_col).copy()
        g["signal_date"] = g[date_col]
        g[date_col] = g[date_col].shift(-sessions)
        out.append(g[g[date_col].notna()])
    if not out:
        return sig.iloc[:0]
    return (pd.concat(out, ignore_index=True)
            .sort_values([date_col, group_col]).reset_index(drop=True))


def gate_report(scored: pd.DataFrame) -> pd.DataFrame:
    """
    How often each gate is the binding constraint.

    This is a headline diagnostic, not an afterthought: it shows WHY the product
    is quiet, and it is the evidence that silence is a decision rather than a bug.
    """
    gates = [c for c in scored.columns if c.startswith("gate_")]
    rows = []
    n = len(scored)
    for gcol in gates:
        rows.append({"gate": gcol, "pass": int(scored[gcol].sum()),
                     "fail": int((~scored[gcol].fillna(False)).sum()),
                     "pass_rate": float(scored[gcol].mean()) if n else np.nan})
    rows.append({"gate": "ALL (signal_ok)", "pass": int(scored["signal_ok"].sum()),
                 "fail": int((~scored["signal_ok"]).sum()),
                 "pass_rate": float(scored["signal_ok"].mean()) if n else np.nan})
    return pd.DataFrame(rows)
