"""
Expected edge of a signal, learned walk-forward: is this gap worth its cost?

The magnitude gate (|z| >= 2) says a gap is unusual. It does not say the gap is big
enough to pay for trading it, and on real data most |z| 2-3 gaps are not: they
close by less than a round trip costs. This module estimates, for each signal, how
many basis points the gap will close, so the backtest (and the live board) can
compare that with the cost of the actual trade before entering.

How it learns
-------------
Every past session on every pair is an example with a known answer: enter the
session after the signal, hold `horizon` sessions, and measure how far the spread
moved in the trade's favour. That answer is only known `horizon + 1` sessions
later, so on any decision date the model may train only on examples whose answer
was already known. The model is refitted at the start of every month on all such
examples, and predicts that month's signals. Nothing from a later date reaches an
earlier decision.

Model: ridge regression on four features (gap size in bp, |z|, the pair's
half-life, and whether the two contracts share an expiry window). Deliberately
simple: about a thousand examples, heavily overlapping, is not enough to trust
anything that can memorise. Until `min_train` examples exist the prediction is
left empty and the gate stays open (the system behaves as it did before), and the
output says which rows were predicted.

The cost side is not learned: the bhavcopy carries no bid/ask, so there is no
observed cost to learn from. The backtest prices each trade with the per-contract,
per-day spread estimate and statutory charges it already uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ("abs_dislocation_bp", "abs_z", "halflife", "maturity_matched")


@dataclass
class EdgeConfig:
    horizon: int = 10            # sessions held for the training outcome
    min_train: int = 150         # examples needed before predicting at all
    train_min_abs_z: float = 1.5 # examples used for training: |z| at least this
    ridge_alpha: float = 1.0
    halflife_cap: float = 60.0   # sessions; long or missing half-lives are capped


def outcomes(pairs: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Per pair and signal date: the realised gain of trading the gap, in bp.

    direction = -sign(z): a spread above its centre is sold, below it is bought.
    Entry at the NEXT session (the settlement is only known after the close), exit
    `horizon` sessions after entry. `known_date` is the session whose close reveals
    the outcome; a model may use the row only on later dates.
    """
    if pairs.empty:
        return pairs.assign(outcome_bp=np.nan, known_date=pd.NaT)
    out = []
    for _pair, g in pairs.sort_values("date").groupby("pair", sort=False):
        g = g.copy()
        s = g["spread_bp"].astype(float).to_numpy()
        d = g["date"].to_numpy()
        n = len(g)
        entry = np.full(n, np.nan)
        exit_ = np.full(n, np.nan)
        known = np.full(n, None, dtype=object)
        idx = np.arange(n)
        ok = idx + 1 + horizon < n
        entry[ok] = s[idx[ok] + 1]
        exit_[ok] = s[idx[ok] + 1 + horizon]
        known[ok] = d[idx[ok] + 1 + horizon]
        direction = -np.sign(g["z"].astype(float).to_numpy())
        g["outcome_bp"] = direction * (exit_ - entry)
        g["known_date"] = known
        out.append(g)
    return pd.concat(out).sort_index()


def _features(df: pd.DataFrame, cfg: EdgeConfig) -> np.ndarray:
    hl = df["halflife"].astype(float).clip(upper=cfg.halflife_cap)
    hl = hl.fillna(cfg.halflife_cap)
    return np.column_stack([
        df["dislocation_bp"].astype(float).abs().to_numpy(),
        df["z"].astype(float).abs().to_numpy(),
        hl.to_numpy(),
        df["maturity_matched"].astype(float).to_numpy(),
    ])


def predict_walk_forward(pairs: pd.DataFrame, cfg: EdgeConfig | None = None,
                         only_signals: bool = True) -> pd.DataFrame:
    """
    Add `exp_gain_bp` (expected bp the gap closes over the horizon) and
    `edge_trained_on` (examples behind the prediction; 0 = not predicted).

    Refits once per calendar month on examples known before that month began.
    With `only_signals`, predicts only rows that pass the signal gates (the only
    rows a trade can come from); training still uses every eligible example.
    """
    from sklearn.linear_model import Ridge

    cfg = cfg or EdgeConfig()
    out = pairs.copy()
    out["exp_gain_bp"] = np.nan
    out["edge_trained_on"] = 0
    if out.empty:
        return out

    ex = outcomes(out, cfg.horizon)
    train_pool = ex[(ex["z"].abs() >= cfg.train_min_abs_z)
                    & ex["outcome_bp"].notna()
                    & ex["dislocation_bp"].notna()].copy()
    train_pool["known_date"] = pd.to_datetime(train_pool["known_date"])

    target = out["signal_ok"].astype(bool) if only_signals else pd.Series(True, index=out.index)
    target &= out["z"].notna() & out["dislocation_bp"].notna()
    if not target.any():
        return out

    dates = pd.to_datetime(out["date"])
    months = dates.dt.to_period("M")
    for m in sorted(months[target].unique()):
        rows = target & (months == m)
        cutoff = m.start_time                       # first day of the month
        train = train_pool[train_pool["known_date"] < cutoff]
        if len(train) < cfg.min_train:
            continue                                # not enough history: no prediction
        model = Ridge(alpha=cfg.ridge_alpha)
        model.fit(_features(train, cfg), train["outcome_bp"].to_numpy())
        out.loc[rows, "exp_gain_bp"] = model.predict(_features(out.loc[rows], cfg))
        out.loc[rows, "edge_trained_on"] = len(train)
    return out


def rule_gain(pairs: pd.DataFrame, exit_z: float = 0.5) -> pd.Series:
    """
    The un-learned alternative, for comparison: assume the gap closes all the way
    to the exit band, i.e. |dislocation| x (1 - exit_z / |z|). Optimistic by
    construction; a learned model has to beat this to earn its place.
    """
    z = pairs["z"].astype(float).abs()
    frac = (1.0 - exit_z / z).clip(lower=0.0)
    return pairs["dislocation_bp"].astype(float).abs() * frac
