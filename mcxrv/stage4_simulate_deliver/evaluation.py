"""
Attribution: prove the P&L is relative value and not something else wearing its
clothes.

Three impostors have to be ruled out, and the brief names all three.

1. GOLD BETA. A spread that is fine-gram neutral has no first-order exposure to
   the gold level, because the level term cancels between the legs. But lot
   indivisibility leaves a residual, so the cancellation is approximate. We
   therefore regress daily strategy return on the model's own gold factor
   (`daily.d_level`) and report beta WITH a confidence interval. Asserting
   neutrality by construction is not evidence; a beta near zero with a tight
   interval is.

2. MECHANICAL ROLL-DOWN. A contract at carry slope b loses b*dt of premium simply
   by ageing. Any position with unequal time-to-expiry across legs harvests this
   automatically. It is arbitrage-free drift, known in advance, and not skill.

3. CURVE RESHAPING. A change in the slope itself moves legs at different
   maturities by different amounts. This is a genuine market move, but it is a
   term-structure bet, not a cross-contract relative-value bet.

The decomposition, in log space, from the model
    ln P[i,t] = a_t + b_t * tau[i,t] + c_s + e[i,t]

For one leg between entry and exit:
    d lnP = da  +  d(b*tau)  +  dc_s  +  de
    d(b*tau) = b_0 * dtau        <- roll-down, mechanical
             + tau_0 * db        <- curve reshaping
             + db * dtau         <- cross term, second order

For a long-a / short-b spread the `da` terms cancel to the extent the legs are
gram-matched, so the level only enters through the residual mismatch:

    level_leak  = da * residual_gram_fraction
    roll        = b_0 * (dtau_a - dtau_b)
    curve       = db  * (tau_a0 - tau_b0)
    alpha       = de_a - de_b        <- the only term that is the strategy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stage1_intake import specs


# ---------------------------------------------------------------------------
# regression helper
# ---------------------------------------------------------------------------

def ols(y: np.ndarray, X: np.ndarray, names=None) -> dict:
    """
    Least squares with heteroskedasticity-robust (HC1) standard errors.

    HC1 rather than classical: daily P&L is heteroskedastic across regimes, and
    classical errors would overstate significance exactly where it matters.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n <= k + 1:
        return {"n": int(n), "error": "insufficient observations"}

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - k
    # HC1
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = xtx_inv @ S @ xtx_inv * (n / max(dof, 1))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan

    names = names or ["x{}".format(i) for i in range(k)]
    return {
        "n": int(n), "r2": r2, "dof": int(dof),
        "params": {nm: float(b) for nm, b in zip(names, beta)},
        "se": {nm: float(s) for nm, s in zip(names, se)},
        "tstat": {nm: float(t) for nm, t in zip(names, tstat)},
        "ci95": {nm: (float(b - 1.96 * s), float(b + 1.96 * s))
                 for nm, b, s in zip(names, beta, se)},
    }


# ---------------------------------------------------------------------------
# 1. gold beta
# ---------------------------------------------------------------------------

def gold_beta(bt_daily: pd.DataFrame, carry_daily: pd.DataFrame,
              return_col: str = "ret") -> dict:
    """
    Regress daily strategy return on the gold factor.

    The factor is `d_level` from the carry fit: the daily change in the fitted log
    level of fine gold. Using the model's own level rather than an external gold
    series keeps the factor internally consistent -- same data, same normalisation,
    no basis introduced by an unrelated reference price.

    Reports beta with a 95% interval and the share of variance gold explains. A
    strategy whose R-squared against gold is high is a gold bet, whatever it is
    called.
    """
    m = bt_daily.merge(carry_daily[["date", "d_level", "d_slope"]], on="date",
                       how="left")
    y = m[return_col].to_numpy(dtype=float)
    gold = m["d_level"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(gold), gold])
    res = ols(y, X, names=["alpha", "beta_gold"])
    if "error" in res:
        return res

    res["interpretation"] = _beta_verdict(res)
    # Annualised alpha, for readability alongside the headline return.
    res["alpha_annualised"] = res["params"]["alpha"] * 252
    return res


def _beta_verdict(res: dict) -> str:
    lo, hi = res["ci95"]["beta_gold"]
    r2 = res.get("r2", np.nan)
    if lo <= 0.0 <= hi:
        base = "gold beta is not distinguishable from zero"
    else:
        base = "gold beta is significantly non-zero: this carries directional gold risk"
    if np.isfinite(r2) and r2 > 0.10:
        base += "; gold explains {:.0%} of return variance".format(r2)
    return base


def two_factor(bt_daily: pd.DataFrame, carry_daily: pd.DataFrame,
               return_col: str = "ret") -> dict:
    """
    Return against BOTH the gold level and the carry slope.

    Separates a directional gold bet from a term-structure bet. Residual alpha
    after both is the part attributable to cross-contract relative value.
    """
    m = bt_daily.merge(carry_daily[["date", "d_level", "d_slope"]], on="date",
                       how="left")
    y = m[return_col].to_numpy(dtype=float)
    X = np.column_stack([
        np.ones(len(m)),
        m["d_level"].to_numpy(dtype=float),
        m["d_slope"].to_numpy(dtype=float),
    ])
    res = ols(y, X, names=["alpha", "beta_gold", "beta_slope"])
    if "error" not in res:
        res["alpha_annualised"] = res["params"]["alpha"] * 252
    return res


# ---------------------------------------------------------------------------
# 2 & 3. per-trade decomposition
# ---------------------------------------------------------------------------

def decompose_trades(trades: pd.DataFrame, rows: pd.DataFrame,
                     carry_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Split each trade's log-space return into level leak, roll-down, curve and alpha.

    Every component is in basis points of one leg's notional, so they are directly
    comparable with the net_bp the backtest reports and with the cost estimates.

    `alpha_bp` is the only component that represents the intended strategy. If
    roll_bp dominates it, the strategy is a calendar-carry harvester that happens
    to be described as relative value.
    """
    if trades.empty:
        return trades

    key = rows.set_index(["date", "contract_id"])
    need = ["residual", "T_years", "log_norm"]
    key = key[[c for c in need if c in key.columns]]
    dl = carry_daily.set_index("date")[["level", "slope"]]

    recs = []
    for tr in trades.itertuples(index=False):
        try:
            a0 = key.loc[(tr.entry_date, tr.cid_a)]
            a1 = key.loc[(tr.exit_date, tr.cid_a)]
            b0 = key.loc[(tr.entry_date, tr.cid_b)]
            b1 = key.loc[(tr.exit_date, tr.cid_b)]
            d0 = dl.loc[tr.entry_date]
            d1 = dl.loc[tr.exit_date]
        except KeyError:
            recs.append({})
            continue

        da = float(d1["level"] - d0["level"])
        db = float(d1["slope"] - d0["slope"])
        b_0 = float(d0["slope"])

        dtau_a = float(a1["T_years"] - a0["T_years"])
        dtau_b = float(b1["T_years"] - b0["T_years"])
        tau_a0 = float(a0["T_years"])
        tau_b0 = float(b0["T_years"])

        de_a = float(a1["residual"] - a0["residual"])
        de_b = float(b1["residual"] - b0["residual"])

        sgn = float(tr.dir_pair)
        # residual gram mismatch: the only channel by which the level reaches P&L
        resid_frac = float(tr.residual_frac) * np.sign(float(tr.fine_residual_g) or 1.0)

        level_leak = da * resid_frac * sgn
        roll = b_0 * (dtau_a - dtau_b) * sgn
        curve = db * (tau_a0 - tau_b0) * sgn
        cross = (db * dtau_a - db * dtau_b) * sgn
        alpha = (de_a - de_b) * sgn

        recs.append({
            "level_leak_bp": level_leak * 10000.0,
            "roll_bp": roll * 10000.0,
            "curve_bp": curve * 10000.0,
            "cross_bp": cross * 10000.0,
            "alpha_bp": alpha * 10000.0,
            "explained_bp": (level_leak + roll + curve + cross + alpha) * 10000.0,
            "d_gold_log": da,
            "d_slope": db,
        })

    dec = pd.DataFrame(recs, index=trades.index)
    out = pd.concat([trades.reset_index(drop=True), dec.reset_index(drop=True)],
                    axis=1)
    if "gross_bp" in out.columns and "explained_bp" in out.columns:
        # How much of realised gross return the decomposition accounts for. A large
        # unexplained residual means the model is missing a factor and the
        # attribution should not be trusted.
        out["unexplained_bp"] = out["gross_bp"] - out["explained_bp"]
    return out


def attribution_summary(dec: pd.DataFrame) -> pd.DataFrame:
    """Mean contribution of each component, with the cost drag beside it."""
    comps = ["alpha_bp", "roll_bp", "curve_bp", "level_leak_bp", "cross_bp",
             "unexplained_bp"]
    rows = []
    for c in comps:
        if c not in dec.columns:
            continue
        v = dec[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        rows.append({
            "component": c.replace("_bp", ""),
            "mean_bp": float(np.mean(v)),
            "sd_bp": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
            "share_of_abs": np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        tot = out["mean_bp"].abs().sum()
        out["share_of_abs"] = (out["mean_bp"].abs() / tot) if tot > 0 else np.nan
    # Costs are not a model component, but the comparison is the whole point.
    if "net_bp" in dec.columns and "gross_bp" in dec.columns:
        cost = float((dec["gross_bp"] - dec["net_bp"]).mean())
        out = pd.concat([out, pd.DataFrame([{
            "component": "cost_drag", "mean_bp": -cost, "sd_bp": np.nan,
            "share_of_abs": np.nan}])], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# significance under overlap
# ---------------------------------------------------------------------------

def block_bootstrap_mean(x, block: int = 10, n_boot: int = 5000,
                         seed: int = 0) -> dict:
    """
    Block bootstrap confidence interval for a mean.

    Overlapping holding periods make daily P&L serially dependent, so an iid
    bootstrap (or a plain t-test) understates the standard error. Resampling
    contiguous blocks preserves the autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block * 3:
        return {"n": int(n), "error": "series too short for block bootstrap"}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(n), "mean": float(x.mean()),
        "ci95": (float(lo), float(hi)),
        "p_two_sided_vs_zero": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
        "block": block, "n_boot": n_boot,
    }


def deflated_note(n_configs_tried: int, best_sharpe: float) -> str:
    """
    A blunt reminder about multiple testing.

    Trying many configurations and reporting the best inflates the apparent Sharpe.
    The expected maximum of n independent zero-mean Sharpe draws grows roughly like
    sqrt(2 ln n) / sqrt(T_years), so a "good" number from a wide search may be
    nothing at all.
    """
    if n_configs_tried <= 1:
        return "single configuration; no selection adjustment needed"
    expected_max = np.sqrt(2.0 * np.log(n_configs_tried))
    return (
        "{} configurations evaluated. The expected maximum |t| from pure noise "
        "across that many trials is about {:.2f}. A reported Sharpe should be "
        "compared against that bar, not against zero."
        .format(n_configs_tried, expected_max)
    )
