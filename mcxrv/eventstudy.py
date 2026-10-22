"""
Event study on the residual spread: does an extreme dislocation actually converge?

Why this exists alongside the backtest. The backtest answers "could I have traded
this?", and after liquidity, lot granularity and calendar gates it answers that
question on a handful of trades -- far too few to establish anything statistically.
Those two questions deserve separating:

  STATISTICAL   Does a wide spread predict its own narrowing? Every pair-day with a
                signal counts, tradable or not. Large N, real inference.
  EXECUTABLE    Of those, how many could be entered at size, held to convergence
                inside the contract calendar, and exited for less than the edge?
                Small N, and the honest answer usually lives here.

Conflating them lets a strong statistical result hide behind a thin backtest, or a
thin backtest discredit a real statistical effect. Reporting both, and showing the
attrition from one to the other, is the actual finding.

Inference detail: consecutive observations at horizon h overlap by h-1 sessions, so
they are not independent. A plain t-test on overlapping forward returns overstates
significance by roughly sqrt(h). Three defences are reported side by side: a
Newey-West corrected t-statistic, a block bootstrap, and a strictly
non-overlapping subsample taken every h-th observation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_Z_BANDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, np.inf))


def _newey_west_tstat(x: np.ndarray, lags: int) -> float:
    """
    t-statistic for a mean, with a Newey-West HAC variance estimate.

    Bartlett kernel. Corrects for the serial correlation that overlapping forward
    windows induce; without it the t-statistic is inflated by roughly sqrt(h).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        gl = float(e[L:] @ e[:-L]) / n
        var += 2.0 * w * gl
    if var <= 0:
        return np.nan
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else np.nan


def _block_bootstrap_ci(x: np.ndarray, block: int, n_boot: int = 3000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max(block * 3, 20):
        return (np.nan, np.nan), np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    hi_start = n - block + 1
    means = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, hi_start, size=nb)
        means[i] = np.concatenate([x[j:j + block] for j in s])[:n].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float(2 * min((means <= 0).mean(), (means >= 0).mean()))
    return (float(lo), float(hi)), p


def forward_convergence(pair_sig: pd.DataFrame, horizons=DEFAULT_HORIZONS,
                        entry_lag: int = 1) -> pd.DataFrame:
    """
    Attach forward signed convergence at each horizon, per pair.

    `conv_{h}_bp` is the P&L in bp of taking the mean-reversion side, entering
    `entry_lag` sessions after the signal, and holding h sessions:

        conv = -sign(z[t]) * (spread[t + lag + h] - spread[t + lag])

    ENTRY_LAG DEFAULTS TO 1, AND MUST NOT BE SET TO 0 FOR INFERENCE. With lag 0 the
    expression contains +|spread[t] - centre|, which is positive by construction, so
    ANY noise in spread[t] manufactures apparent convergence -- conditioning on a
    large deviation guarantees the next draw sits closer to the mean. On zero-edge
    synthetic data lag 0 reports t-statistics above 30 for an effect that does not
    exist. Entering at t+1 breaks the shared noise term between the signal and the
    entry price.

    Lag 1 is also the only honest execution assumption: a settlement is published
    after the close, so a signal computed from it cannot be filled at it.

    `conv_lag0_{h}_bp` is retained purely as a diagnostic, to show the size of the
    bias that lag 0 would have introduced.
    """
    parts = []
    for pair, g in pair_sig.sort_values("date").groupby("pair"):
        g = g.reset_index(drop=True).copy()
        s = g["spread_bp"].astype(float)
        sgn = -np.sign(g["z"].fillna(0.0))
        entry = s.shift(-entry_lag)
        for h in horizons:
            fwd = s.shift(-(entry_lag + h)) - entry
            g["conv_{}_bp".format(h)] = sgn * fwd
            g["conv_lag0_{}_bp".format(h)] = sgn * (s.shift(-h) - s)
            # Realised absolute move, for a sanity check on volatility scaling.
            g["absmove_{}_bp".format(h)] = fwd.abs()
        g["pair"] = pair
        g["entry_lag"] = entry_lag
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def study(pair_sig: pd.DataFrame, horizons=DEFAULT_HORIZONS,
          z_bands=DEFAULT_Z_BANDS, require_gates: bool = False,
          maturity_matched_only: bool = False,
          min_history_only: bool = True, entry_lag: int = 1) -> pd.DataFrame:
    """
    Mean forward convergence by |z| band and horizon, with overlap-aware inference.

    `require_gates=False` by default: the statistical question is whether the
    dislocation predicts convergence, which should be asked of every observation
    rather than only the ones that happened to clear a liquidity filter. Filtering
    first would condition the sample on tradability and bias the estimate.

    The |z| < 1 band is the control. If the extreme bands do not beat it, there is
    no signal -- only volatility, which rises with |z| mechanically.
    """
    fc = forward_convergence(pair_sig, horizons, entry_lag=entry_lag)
    if fc.empty:
        return pd.DataFrame()

    if min_history_only and "gate_history" in fc.columns:
        fc = fc[fc["gate_history"].fillna(False)]
    if require_gates and "signal_ok" in fc.columns:
        fc = fc[fc["signal_ok"]]
    if maturity_matched_only and "maturity_matched" in fc.columns:
        fc = fc[fc["maturity_matched"]]

    rows = []
    az = fc["z"].abs()
    for lo, hi in z_bands:
        band = fc[(az >= lo) & (az < hi)]
        label = "|z| {:.0f}-{}".format(lo, "inf" if not np.isfinite(hi) else "{:.0f}".format(hi))
        for h in horizons:
            col = "conv_{}_bp".format(h)
            x = band[col].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < 10:
                continue
            nw = _newey_west_tstat(x, lags=max(h, 1))
            (blo, bhi), bp_ = _block_bootstrap_ci(x, block=max(h + 5, 10))
            # Strictly non-overlapping subsample.
            xn = x[::max(h, 1)]
            t_nonover = (float(np.mean(xn) / (np.std(xn, ddof=1) / np.sqrt(len(xn))))
                         if len(xn) > 5 and np.std(xn, ddof=1) > 0 else np.nan)
            # Size of the bias that same-bar entry would have introduced.
            x0 = band["conv_lag0_{}_bp".format(h)].to_numpy(dtype=float)
            x0 = x0[np.isfinite(x0)]
            lag0_mean = float(np.mean(x0)) if len(x0) else np.nan

            rows.append({
                "z_band": label, "horizon": h, "n": int(len(x)),
                "mean_conv_bp": float(np.mean(x)),
                "median_conv_bp": float(np.median(x)),
                "sd_bp": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "hit_rate": float((x > 0).mean()),
                "t_naive": float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))
                if np.std(x, ddof=1) > 0 else np.nan,
                "t_newey_west": nw,
                "t_nonoverlap": t_nonover,
                "n_nonoverlap": int(len(xn)),
                "boot_ci_lo": blo, "boot_ci_hi": bhi, "boot_p": bp_,
                "mean_conv_lag0_bp": lag0_mean,
                "lag0_bias_bp": lag0_mean - float(np.mean(x)),
            })
    return pd.DataFrame(rows)


def cost_hurdle_table(es: pd.DataFrame, round_trip_bp: float) -> pd.DataFrame:
    """
    Net the measured convergence against a round-trip cost, per band and horizon.

    This is the table that decides the project. Gross convergence can be
    statistically bulletproof and still sit below the cost line, which is the
    outcome the brief explicitly accepts as valid.
    """
    if es.empty:
        return es
    out = es.copy()
    out["round_trip_bp"] = round_trip_bp
    out["net_conv_bp"] = out["mean_conv_bp"] - round_trip_bp
    out["survives_costs"] = out["net_conv_bp"] > 0
    # Does the bootstrap interval clear the hurdle, not merely clear zero?
    out["ci_clears_costs"] = out["boot_ci_lo"] > round_trip_bp
    return out[["z_band", "horizon", "n", "mean_conv_bp", "round_trip_bp",
                "net_conv_bp", "hit_rate", "t_newey_west", "boot_ci_lo",
                "boot_ci_hi", "survives_costs", "ci_clears_costs"]]


def by_pair(pair_sig: pd.DataFrame, z_min: float = 2.0,
            horizon: int = 10) -> pd.DataFrame:
    """
    Per-pair breakdown at one band and horizon.

    Pairs differ enormously in history length and liquidity -- GOLDTEN only lists
    from 2025 -- so a pooled number can be carried entirely by one pair. This table
    is how that gets caught, and it reports `years_history` so a short sample cannot
    masquerade as evidence.
    """
    fc = forward_convergence(pair_sig, (horizon,), entry_lag=1)
    if fc.empty:
        return pd.DataFrame()
    col = "conv_{}_bp".format(horizon)
    rows = []
    for pair, g in fc.groupby("pair"):
        sub = g[(g["z"].abs() >= z_min) & g[col].notna()]
        if "gate_history" in g.columns:
            sub = sub[sub["gate_history"].fillna(False)]
        if len(sub) < 10:
            rows.append({"pair": pair, "n": int(len(sub)),
                         "note": "too few observations"})
            continue
        x = sub[col].to_numpy(dtype=float)
        dates = pd.to_datetime(g["date"])
        rows.append({
            "pair": pair,
            "maturity_matched": bool(g["maturity_matched"].iloc[0]),
            "n": int(len(x)),
            "years_history": round((dates.max() - dates.min()).days / 365.25, 2),
            "mean_conv_bp": round(float(np.mean(x)), 2),
            "hit_rate": round(float((x > 0).mean()), 3),
            "t_newey_west": round(_newey_west_tstat(x, lags=horizon), 2),
            "median_halflife": round(float(sub["halflife"].median()), 1)
            if "halflife" in sub.columns else np.nan,
        })
    return pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)
