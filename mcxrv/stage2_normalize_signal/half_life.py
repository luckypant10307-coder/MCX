"""
Term-structure model: separate the level, the carry curve, the structural premium
and the relative-value residual.

Model, in log space, for contract i of symbol s on day t:

    ln P_norm[i,t] = a_t  +  b_t * tau[i,t]  +  c_s  +  e[i,t]

    a_t   log level of fine gold. Daily. THIS is the gold factor -- its first
          difference is the gold return used for beta attribution, and it is
          derived from the same cross-section as everything else rather than
          borrowed from an unrelated series.
    b_t   annualised carry slope. Daily. Generates roll-down mechanically: as
          tau falls, the carry premium decays to zero whether or not anything
          in the market changes. A calendar spread harvests this; it is not edge.
    c_s   structural log premium of symbol s. Fabrication cost embedded in a
          minted deliverable, delivery-centre and participant-base differences.
          PERSISTENT, and the thing a naive parity spread mistakes for mispricing.
    e[i,t] residual. The only term that is a candidate relative-value signal.

IDENTIFICATION, and why it also enforces no look-ahead
------------------------------------------------------
Within a single day, c_s and a symbol-level dislocation are the same object: both
are a constant applied to every contract of that symbol. They are separable only
through time, by persistence -- c_s is stable, e[s,t] mean-reverts. So:

  * c_s is estimated ONLY from a trailing window ending at t-1.
  * a_t and b_t are fit on day t's cross-section alone, which is information
    genuinely available at t.

The consequence is that the no-look-ahead property is structural, not a discipline
applied afterwards. There is no code path by which a future observation reaches
the residual for day t.

c_s is identified up to an additive constant; GOLDM is pinned to zero, so a_t is
GOLDM's own level and every premium is quoted relative to the 995 bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REF_SYMBOL = "GOLDM"


@dataclass
class CarryConfig:
    window_days: int = 120        # trailing window for the structural premiums
    refit_every: int = 5          # recalibrate premiums weekly, not daily
    min_obs_per_day: int = 4      # need a cross-section to fit a line
    min_tau_spread: float = 0.02  # need maturity dispersion (~5 sessions)
    iters: int = 6                # alternating projections; converges fast
    min_window_days: int = 40     # refuse to emit signal before this much history
    exclude_derived_from_fit: bool = True
    tender_weight: float = 0.25   # downweight, do not exclude, the tender window

    # --- robustness to a single bad print -------------------------------------
    # Plain weighted least squares gives one corrupt settlement unbounded influence
    # on that day's slope, and that error propagates into EVERY residual for the
    # session. Huber loss bounds it: quadratic for small residuals, linear beyond
    # `huber_epsilon` standardised deviations.
    #
    # `robust_scope` controls where the cost is paid, because the two fits are called
    # very different numbers of times:
    #   "day"  same-day level/slope fit only (~1 call per session). This is where a
    #          bad print does real damage, so it is the default.
    #   "all"  also every fit inside the trailing premium sweep (~700 calls per
    #          refit). Thorough but much slower for little gain, since the premium
    #          is already an average over roughly a thousand observations.
    #   "none" pure weighted least squares everywhere.
    #
    # Measured on synthetic data with one 300 bp corrupt print in the cross-section:
    # WLS mis-states the level by +23.9 bp and the slope by +0.0092; Huber by
    # +0.09 bp and +0.0003. On clean data the two agree to 0.02 bp, so robustness
    # costs nothing when there is nothing to resist. Full-pipeline cost: ~8% slower.
    robust_scope: str = "day"
    huber_epsilon: float = 1.35   # 95% Gaussian efficiency; must exceed 1.0
    huber_alpha: float = 1e-6     # tiny L2, for when the maturity spread is thin
    huber_max_iter: int = 200

    # Aggregate the premium update with a weighted median instead of a weighted mean.
    # OFF by default: it was tried and did not earn its place. c_s is already an
    # average over ~120 days x ~10 contracts, so a single bad print shifts the mean by
    # about 1/1200 of its size -- already negligible. Measured premium recovery error
    # was 5.88 bp with the mean and 6.02 bp with the median, i.e. no better. Kept as
    # an option, not a default.
    robust_premium_median: bool = False


def fit_weights(df: pd.DataFrame, cfg: CarryConfig) -> pd.Series:
    """
    Regression weights. Liquidity-proportional, because the curve should be set by
    prices someone actually traded.

    Derived settlements get zero weight: they are the exchange's own smooth model
    output and would drag the fitted curve toward itself, shrinking exactly the
    residual we are trying to measure.
    """
    w = np.log1p(df["volume"].clip(lower=0).astype(float))
    if cfg.exclude_derived_from_fit:
        w = w.where(~df["settle_untradeable"], 0.0)
    if "in_tender" in df.columns:
        w = w.where(~df["in_tender"], w * cfg.tender_weight)
    return w.fillna(0.0)


_HUBER_AVAILABLE = None


def _huber_available() -> bool:
    """scikit-learn is optional: the model degrades to WLS without it."""
    global _HUBER_AVAILABLE
    if _HUBER_AVAILABLE is None:
        try:
            from sklearn.linear_model import HuberRegressor  # noqa: F401
            _HUBER_AVAILABLE = True
        except ImportError:
            _HUBER_AVAILABLE = False
    return _HUBER_AVAILABLE


def _fit_day(tau: np.ndarray, y: np.ndarray, w: np.ndarray,
             robust: bool = False, cfg: "CarryConfig | None" = None):
    """
    Fit y on [1, tau]. Returns (a, b, rmse, n).

    Weighted least squares by default. With `robust=True` and scikit-learn present,
    Huber loss instead, which bounds the influence of a single corrupt settlement on
    the slope. Falls back to WLS silently whenever Huber is unavailable, the problem
    is too small, or the solver fails to converge -- a robustness feature that could
    itself drop a day's fit would be a downgrade.

    The reported `rmse` is always the weighted least-squares residual RMSE about the
    fitted line, so it stays comparable across both paths and continues to mean the
    same thing to `liquidity.annotate`, which uses it as the measurement noise floor.
    """
    ok = (w > 0) & np.isfinite(y) & np.isfinite(tau)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan, int(ok.sum())
    t, yy, ww = tau[ok], y[ok], w[ok]
    if np.ptp(t) < 1e-9:
        return np.nan, np.nan, np.nan, int(ok.sum())

    a = b = None
    if robust and _huber_available() and ok.sum() >= 5:
        cfg = cfg or CarryConfig()
        from sklearn.linear_model import HuberRegressor
        # Centre y before fitting. Log prices sit near 10.8 while tau spans ~0-0.5,
        # so an uncentred intercept is numerically dominant and makes Huber's joint
        # scale estimate badly conditioned.
        y0 = float(np.average(yy, weights=ww))
        try:
            hr = HuberRegressor(epsilon=cfg.huber_epsilon, alpha=cfg.huber_alpha,
                                max_iter=cfg.huber_max_iter, fit_intercept=True)
            hr.fit(t.reshape(-1, 1), yy - y0, sample_weight=ww)
            a = float(hr.intercept_) + y0
            b = float(hr.coef_[0])
            if not (np.isfinite(a) and np.isfinite(b)):
                a = b = None
        except (ValueError, FloatingPointError):
            a = b = None

    if a is None:
        X = np.column_stack([np.ones_like(t), t])
        sw = np.sqrt(ww)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], yy * sw, rcond=None)
        a, b = float(beta[0]), float(beta[1])

    resid = yy - (a + b * t)
    rmse = float(np.sqrt(np.average(resid ** 2, weights=ww)))
    return a, b, rmse, int(ok.sum())


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: the L1 centre, unmoved by an extreme tail observation."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return 0.0
    v, w = v[keep], w[keep]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    half = 0.5 * cum[-1]
    idx = int(np.searchsorted(cum, half))
    return float(v[min(idx, len(v) - 1)])


def _estimate_premiums(win: pd.DataFrame, cfg: CarryConfig) -> dict:
    """
    Structural premiums c_s from a trailing window, by alternating projections.

    Iterate: fit each day's (a, b) holding c fixed, then update each symbol's c by
    its weighted mean residual across the window. This is the standard two-way
    fixed-effects sweep and converges in a handful of passes.
    """
    syms = list(win["symbol"].unique())
    c = {s: 0.0 for s in syms}
    if win.empty:
        return c

    days = [(d, g) for d, g in win.groupby("date", sort=True)
            if len(g) >= cfg.min_obs_per_day]

    for _ in range(cfg.iters):
        parts = []
        for _d, g in days:
            adj = g["symbol"].map(c).astype(float).to_numpy()
            y = g["log_norm"].to_numpy() - adj
            tau = g["T_years"].to_numpy()
            w = g["_w"].to_numpy()
            live = w > 0
            if not live.any() or np.ptp(tau[live]) < cfg.min_tau_spread:
                continue
            a, b, _rmse, n = _fit_day(tau, y, w,
                                      robust=(cfg.robust_scope == "all"), cfg=cfg)
            if not np.isfinite(a):
                continue
            r = y - (a + b * tau)
            parts.append(pd.DataFrame({"symbol": g["symbol"].to_numpy(),
                                       "r": r, "w": w}))
        if not parts:
            break
        R = pd.concat(parts, ignore_index=True)
        R = R[R["w"] > 0]
        if R.empty:
            break
        if cfg.robust_premium_median:
            agg = lambda g: _weighted_median(g["r"].to_numpy(), g["w"].to_numpy())
        else:
            agg = lambda g: float(np.average(g["r"], weights=g["w"]))
        upd = R.groupby("symbol").apply(agg, include_groups=False)
        for s, v in upd.items():
            c[s] = c.get(s, 0.0) + float(v)
        # Pin the reference so the premiums are interpretable and a_t is GOLDM.
        if REF_SYMBOL in c:
            base = c[REF_SYMBOL]
            for s in list(c):
                c[s] -= base
    return c


def fit(df: pd.DataFrame, cfg: CarryConfig | None = None):
    """
    Walk forward day by day, emitting the decomposition.

    Returns (rows, daily, premiums):
      rows      input frame plus fitted_log, residual, residual_bp, fit_weight,
                and `in_fit` (whether the row informed that day's curve)
      daily     per-date level a_t, slope b_t, rmse, n_obs, d_level (gold return)
      premiums  per-date per-symbol c_s actually in force that day -- a trailing
                estimate, so it is auditable as of-date rather than final-sample
    """
    cfg = cfg or CarryConfig()
    d = df.copy()
    d["_w"] = fit_weights(d, cfg)

    dates = sorted(d["date"].unique())
    by_date = {k: g for k, g in d.groupby("date", sort=True)}

    prem_cache: dict = {}
    last_refit_idx = None

    row_parts, daily_rows, prem_rows = [], [], []

    for i, t in enumerate(dates):
        g = by_date[t]

        # --- trailing window, strictly before t ---
        lo = max(0, i - cfg.window_days)
        win_dates = dates[lo:i]
        enough = len(win_dates) >= cfg.min_window_days

        if enough and (last_refit_idx is None
                       or (i - last_refit_idx) >= cfg.refit_every):
            win = pd.concat([by_date[u] for u in win_dates], ignore_index=True)
            prem_cache = _estimate_premiums(win, cfg)
            last_refit_idx = i

        c = prem_cache if enough else {}

        # --- same-day level and slope, premiums held fixed ---
        adj = g["symbol"].map(lambda s: c.get(s, 0.0)).astype(float).to_numpy()
        y = g["log_norm"].to_numpy() - adj
        tau = g["T_years"].to_numpy()
        w = g["_w"].to_numpy()
        a, b, rmse, n = _fit_day(
            tau, y, w, robust=(cfg.robust_scope in ("day", "all")), cfg=cfg)

        gg = g.copy()
        if np.isfinite(a):
            gg["fitted_log"] = a + b * tau + adj
            gg["residual"] = gg["log_norm"] - gg["fitted_log"]
        else:
            gg["fitted_log"] = np.nan
            gg["residual"] = np.nan
        gg["residual_bp"] = gg["residual"] * 10000.0
        gg["fit_weight"] = w
        gg["in_fit"] = w > 0
        gg["premium_bp"] = adj * 10000.0
        gg["has_premium_est"] = enough
        row_parts.append(gg)

        daily_rows.append({"date": t, "level": a, "slope": b, "fit_rmse": rmse,
                           "n_obs": n, "n_window_days": len(win_dates),
                           "has_premium_est": enough,
                           "robust_fit": cfg.robust_scope in ("day", "all")
                           and _huber_available()})
        for s, v in (c or {}).items():
            prem_rows.append({"date": t, "symbol": s, "premium_log": v,
                              "premium_bp": v * 10000.0})

    rows = pd.concat(row_parts, ignore_index=True).drop(columns=["_w"])
    daily = pd.DataFrame(daily_rows)
    # d_level is the model's own gold return: the factor we must prove we are
    # NOT simply betting on.
    daily["d_level"] = daily["level"].diff()
    daily["d_slope"] = daily["slope"].diff()
    premiums = pd.DataFrame(prem_rows)
    return rows, daily, premiums


def symbol_residual(rows: pd.DataFrame, weight: str = "fit_weight") -> pd.DataFrame:
    """
    Collapse contract residuals to one dislocation per (date, symbol).

    A whole contract family goes rich or cheap together, so the symbol-level
    aggregate is the cleaner signal carrier than any single expiry. Weighted by
    liquidity, and computed only over rows that informed the fit -- a derived
    settlement cannot tell us about today's dislocation because it does not
    contain today's dislocation.
    """
    r = rows[rows["in_fit"] & rows["residual"].notna()]
    if r.empty:
        return pd.DataFrame(columns=["date", "symbol", "resid_bp", "n", "wsum"])

    def agg(g):
        w = g[weight].to_numpy()
        v = g["residual_bp"].to_numpy()
        if w.sum() <= 0:
            return pd.Series({"resid_bp": np.nan, "n": len(g), "wsum": 0.0})
        return pd.Series({"resid_bp": float(np.average(v, weights=w)),
                          "n": len(g), "wsum": float(w.sum())})

    out = (r.groupby(["date", "symbol"]).apply(agg, include_groups=False)
           .reset_index())
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def roll_down_bp(slope: float, tau_years: float, holding_sessions: int) -> float:
    """
    Expected mechanical roll-down over a holding period, in basis points.

    A contract at slope b and maturity tau loses b * dt of carry premium simply by
    ageing. This is the number that must be netted out before calling a calendar
    position profitable -- it is arbitrage-free drift, not skill.
    """
    dt = holding_sessions / 252.0
    return -slope * min(dt, tau_years) * 10000.0
