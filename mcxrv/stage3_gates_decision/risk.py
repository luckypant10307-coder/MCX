"""
The live risk board: per-pair capital state, and whether to add exposure today.

WHAT THIS IS FOR
----------------
The ledger in `ledger.py` is retrospective -- it records what each position cost to
hold, after the fact. This module answers the question a desk actually asks before
trading: *given what the book already holds and what margin costs this morning,
should this pair be added to?*

That is a different computation, and three properties make it so:

  1. MARGINAL, NOT STANDALONE. The cost of adding a pair is the book's requirement
     WITH it minus the requirement WITHOUT it. SPAN nets deltas and grants
     inter-commodity credit across the whole portfolio, so a pair priced alone can
     look expensive and cost almost nothing at the margin -- or look cheap and cost
     full freight because the credit it was relying on is already spent.

  2. THE REQUIREMENT MOVES. Scan ranges are revised as volatility changes, and the
     delivery add-on switches on as a leg approaches tender. The board therefore
     reports the CHANGE in requirement over the trailing revision cycle, holding the
     position fixed so the number is a statement about risk rather than about sizing.

  3. FREE CASH IS THE BINDING CONSTRAINT. Indian clearing settles MTM in cash daily,
     so a pair that would leave the account with a thin buffer is not addable even
     when the signal is strong and the margin is nominally affordable.

THE VERDICT IS A REFUSAL MECHANISM
----------------------------------
`action` exists to say NO. A signal being statistically valid is necessary and not
sufficient: the position must also be fundable, and it must be fundable under a
deterioration that has not happened yet. Where the board blocks a trade it names the
binding reason, so the refusal can be argued with rather than merely obeyed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stage4_simulate_deliver import replay_driver as bt_mod
from . import margin_capital as margin_mod


@dataclass
class RiskBoardConfig:
    account_capital: float = 1_000_000.0
    # Free cash beyond the requirement, as a multiple of it, below which the buffer
    # is called "low". Matches LedgerConfig.target_buffer_mult.
    target_buffer_mult: float = 0.25
    # Sessions back for the requirement-change comparison. Defaults to one parameter
    # revision cycle, because comparing against yesterday mostly measures noise.
    lookback_sessions: int = 21
    # A rise in requirement of this much over the cycle blocks new exposure: the
    # regime is deteriorating and today's margin is not tomorrow's.
    span_rise_block_pct: float = 0.10
    span_rise_warn_pct: float = 0.05
    # Utilisation of account capital at which adding is discouraged / refused.
    utilisation_warn: float = 0.50
    utilisation_block: float = 0.80
    # Runway a new position needs, in sessions, before the calendar forces it out.
    min_sessions_to_expiry: int = 5
    # Account kill switch: no new exposure once the account is this fraction of
    # account_capital below its peak. Checked before every other rule, because it is
    # a statement about the account, not about the pair.
    max_drawdown_pct: float = 0.10


_BUFFER_ORDER = {"breached": 0, "low": 1, "adequate": 2, "comfortable": 3}


def _buffer_state(free_cash: float, required: float, cfg: RiskBoardConfig) -> str:
    if required <= 0:
        return "comfortable"
    if free_cash < 0:
        return "breached"
    mult = free_cash / required
    if mult < cfg.target_buffer_mult:
        return "low"
    if mult < cfg.target_buffer_mult * 3:
        return "adequate"
    return "comfortable"


def _pick_contract(cands, day, symbol, need_sessions):
    for cid, s2e in cands.get((day, symbol), ()):
        if s2e >= need_sessions:
            return cid, s2e
    return None, None


def _mp(rec, symbol, lots):
    """A margin position from a row record, with whatever risk inputs it carries."""
    sig = getattr(rec, "sigma_daily", None)
    oi = getattr(rec, "open_interest", None)
    return margin_mod.MarginPosition(
        symbol=symbol, contract_id=rec.contract_id, lots=int(lots),
        price=float(rec.close), sessions_to_expiry=int(rec.sessions_to_expiry),
        sigma_daily=float(sig) if sig is not None and np.isfinite(sig) else None,
        open_interest=float(oi) if oi is not None and np.isfinite(oi) else None)


def open_book_from_trades(rows: pd.DataFrame, trades: pd.DataFrame,
                          asof) -> list:
    """
    The positions still open at `asof`, as margin positions marked at that session.

    Reconstructed from the trade blotter rather than carried out of the backtest, so
    the board can be run against any stored blotter -- including one from a previous
    model run -- without re-running the walk-forward.
    """
    if trades is None or trades.empty or rows is None or rows.empty:
        return []
    need = {"entry_date", "exit_date", "cid_a", "cid_b", "lots_a", "lots_b",
            "dir_pair", "sym_a", "sym_b"}
    if not need.issubset(trades.columns):
        return []
    lut = {(r.date, r.contract_id): r for r in rows.itertuples(index=False)}
    live = trades[(trades["entry_date"] <= asof) & (trades["exit_date"] > asof)]
    out = []
    for t in live.itertuples(index=False):
        for cid, sym, lots, sign in ((t.cid_a, t.sym_a, t.lots_a, t.dir_pair),
                                     (t.cid_b, t.sym_b, t.lots_b, -t.dir_pair)):
            rec = lut.get((asof, cid))
            if rec is None:
                continue
            out.append(_mp(rec, sym, int(sign) * int(lots)))
    return out


def board(rows: pd.DataFrame, pair_sig: pd.DataFrame,
          cfg: RiskBoardConfig | None = None,
          span_params: margin_mod.SpanParameters | None = None,
          margin_schedule: dict | None = None,
          bt_cfg: "bt_mod.BacktestConfig | None" = None,
          open_book: list | None = None,
          asof=None,
          account_drawdown_inr: float | None = None) -> pd.DataFrame:
    """
    One row per pair, for a single session: margin, utilisation, buffer, verdict.

    `open_book` is the list of `MarginPosition` already held. It is what makes the
    marginal cost marginal; passing an empty book prices each pair as though it were
    the only thing in the account, which is the optimistic case and is labelled as
    such by `book_required` being zero.
    """
    cfg = cfg or RiskBoardConfig()
    bt_cfg = bt_cfg or bt_mod.BacktestConfig()
    base_params = span_params or margin_mod.SpanParameters()
    if rows is None or rows.empty or pair_sig is None or pair_sig.empty:
        return pd.DataFrame()

    dates = sorted(pd.Series(rows["date"].unique()))
    asof = asof if asof is not None else max(pair_sig["date"])
    if asof not in set(dates):
        return pd.DataFrame()
    i_now = dates.index(asof)
    prior = dates[max(0, i_now - cfg.lookback_sessions)]

    trad = rows[rows.get("tradable", True)] if "tradable" in rows.columns else rows
    cands: dict = {}
    if not trad.empty:
        cols = ["date", "symbol", "contract_id", "volume", "sessions_to_expiry"]
        for rec in (trad[cols].sort_values(["date", "symbol", "volume"],
                                           ascending=[True, True, False])
                    .itertuples(index=False)):
            cands.setdefault((rec.date, rec.symbol), []).append(
                (rec.contract_id, int(rec.sessions_to_expiry)))

    lut = {(r.date, r.contract_id): r
           for r in rows.itertuples(index=False)}

    p_now = margin_mod.params_for_session(base_params, asof, margin_schedule)
    p_prior = margin_mod.params_for_session(base_params, prior, margin_schedule)

    book = list(open_book or [])
    m_book = margin_mod.span_margin(book, p_now)
    book_req = float(m_book["initial_margin"])

    sig_now = pair_sig[pair_sig["date"] == asof]
    out = []
    for rec in sig_now.itertuples(index=False):
        sym_a, sym_b = rec.pair.split("/")
        cid_a, s2e_a = _pick_contract(cands, asof, sym_a, cfg.min_sessions_to_expiry)
        cid_b, s2e_b = _pick_contract(cands, asof, sym_b, cfg.min_sessions_to_expiry)
        if cid_a is None or cid_b is None:
            out.append({"pair": rec.pair, "date": asof, "action": "unavailable",
                        "reason": "no tradable contract with runway",
                        "z": float(getattr(rec, "z", np.nan))})
            continue
        ra, rb = lut.get((asof, cid_a)), lut.get((asof, cid_b))
        if ra is None or rb is None:
            continue

        sized = bt_mod._size_legs(
            sym_a, sym_b, float(ra.close), float(rb.close),
            float(getattr(ra, "max_lots", 0.0)), float(getattr(rb, "max_lots", 0.0)),
            bt_cfg)
        if sized is None:
            out.append({"pair": rec.pair, "date": asof, "action": "unavailable",
                        "reason": "lot granularity cannot hedge to tolerance",
                        "z": float(getattr(rec, "z", np.nan))})
            continue
        lots_a, lots_b, resid_g, resid_frac = sized

        # Short the rich leg: a positive spread z means A is rich to B.
        z = float(getattr(rec, "z", np.nan))
        direction = -1 if (np.isfinite(z) and z > 0) else 1

        legs = [_mp(ra, sym_a, direction * lots_a),
                _mp(rb, sym_b, -direction * lots_b)]
        m_with = margin_mod.span_margin(book + legs, p_now)
        req_with = float(m_with["initial_margin"])
        marginal = req_with - book_req

        # Same position, previous regime: prices and scan ranges as they were a
        # revision cycle ago. Holding lots fixed makes this a risk measurement.
        ra_p, rb_p = lut.get((prior, cid_a)), lut.get((prior, cid_b))
        if ra_p is not None and rb_p is not None:
            legs_p = [_mp(ra_p, sym_a, direction * lots_a),
                      _mp(rb_p, sym_b, -direction * lots_b)]
            m_prior = margin_mod.span_margin(legs_p, p_prior)
        else:
            m_prior = None
        m_alone = margin_mod.span_margin(legs, p_now)

        def _chg(key):
            if m_prior is None:
                return np.nan
            was = float(m_prior[key])
            if was <= 0:
                return np.nan
            return float(m_alone[key]) / was - 1.0

        span_chg = _chg("span")
        im_chg = _chg("initial_margin")

        equity = cfg.account_capital
        util_now = book_req / equity if equity > 0 else np.nan
        util_with = req_with / equity if equity > 0 else np.nan
        free_with = equity - req_with
        state = _buffer_state(free_with, req_with, cfg)

        scan_a = p_now.price_scan_range_pct.get(sym_a, np.nan)
        scan_a_prior = p_prior.price_scan_range_pct.get(sym_a, np.nan)

        signal_ok = bool(getattr(rec, "signal_ok", False))
        runway = int(min(s2e_a, s2e_b))

        # ---- verdict, first binding reason wins ----
        dd_limit = cfg.max_drawdown_pct * cfg.account_capital
        if account_drawdown_inr is not None and account_drawdown_inr >= dd_limit:
            action, reason = ("do not add exposure",
                              "account drawdown {:.1f}% of capital, limit {:.0f}%: "
                              "new exposure halted until reviewed".format(
                                  100 * account_drawdown_inr / cfg.account_capital,
                                  100 * cfg.max_drawdown_pct))
        elif free_with < 0:
            action, reason = "do not add exposure", "requirement exceeds capital"
        elif state == "low":
            action, reason = "do not add exposure", "maintenance buffer low"
        elif np.isfinite(util_with) and util_with >= cfg.utilisation_block:
            action, reason = "do not add exposure", "utilisation above limit"
        elif np.isfinite(span_chg) and span_chg >= cfg.span_rise_block_pct:
            action, reason = ("do not add exposure",
                              "SPAN requirement rising; margin regime deteriorating")
        elif not signal_ok:
            action, reason = "no signal", "gates not cleared"
        elif np.isfinite(util_with) and util_with >= cfg.utilisation_warn:
            action, reason = "add reduced size", "utilisation elevated"
        elif np.isfinite(span_chg) and span_chg >= cfg.span_rise_warn_pct:
            action, reason = "add reduced size", "requirement drifting up"
        else:
            action, reason = "may add", "fundable and gated clear"

        out.append({
            "pair": rec.pair, "date": asof,
            # The contracts actually priced: the most-traded one per leg with enough
            # runway, which is often NOT the front month near expiry.
            "contract_a": cid_a, "contract_b": cid_b,
            "sessions_a": int(s2e_a), "sessions_b": int(s2e_b),
            "lots_a": int(lots_a), "lots_b": int(lots_b), "direction": direction,
            "notional_exposure": float(m_alone["notional"]),
            "residual_frac": float(resid_frac),

            "span_margin": float(m_alone["span"]),
            "initial_margin": float(m_alone["initial_margin"]),
            "elm": float(m_alone["elm"]),
            "delivery_margin": float(m_alone["delivery"]),
            "concentration_margin": float(m_alone["concentration"]),
            "inter_commodity_credit": float(m_alone["inter_credit"]),
            "floor_binds": bool(m_alone["floor_binds"]),
            "marginal_margin": float(marginal),

            "book_required": book_req,
            "required_if_added": req_with,
            "free_cash_if_added": float(free_with),
            "margin_utilisation": float(util_now) if np.isfinite(util_now) else np.nan,
            "utilisation_if_added": float(util_with) if np.isfinite(util_with) else np.nan,
            "buffer_mult": float(free_with / req_with) if req_with > 0 else np.nan,
            "buffer_state": state,

            "span_change_pct": span_chg,
            "im_change_pct": im_chg,
            "scan_range_pct": float(scan_a) if np.isfinite(scan_a) else np.nan,
            "scan_range_prior_pct": float(scan_a_prior)
            if np.isfinite(scan_a_prior) else np.nan,
            "lookback_from": prior,

            "z": z,
            "spread_bp": float(getattr(rec, "spread_bp", np.nan)),
            "signal_ok": signal_ok,
            "sessions_to_expiry": runway,
            "action": action, "reason": reason,
        })

    df = pd.DataFrame(out)
    if df.empty:
        return df
    # A session where every pair is "unavailable" (all contracts inside the expiry
    # runway) has none of the margin columns; add them so the sort below still works.
    for col in ("buffer_state", "utilisation_if_added"):
        if col not in df.columns:
            df[col] = np.nan
    # Worst state first: what a desk needs to see is what is blocked, not what is fine.
    df["_ord"] = df["buffer_state"].map(_BUFFER_ORDER).fillna(9)
    df = df.sort_values(["_ord", "utilisation_if_added"],
                        ascending=[True, False]).drop(columns="_ord")
    return df.reset_index(drop=True)


def alerts(bd: pd.DataFrame, only_blocked: bool = False) -> list[str]:
    """
    The board as the lines a risk monitor would show.

    Each block names the pair, where its capital stands, how the requirement has
    moved, and the resulting instruction -- in that order, because the instruction is
    only credible once the numbers behind it are visible.
    """
    if bd is None or bd.empty:
        return []
    lines = []
    for r in bd.itertuples(index=False):
        if only_blocked and r.action not in ("do not add exposure", "unavailable"):
            continue
        if r.action == "unavailable":
            lines.append("{} spread trade\n  Unavailable: {}".format(r.pair, r.reason))
            continue
        chg = ""
        if np.isfinite(r.span_change_pct):
            verb = "increased" if r.span_change_pct >= 0 else "decreased"
            chg = "\n  SPAN requirement {} by {:.0%}".format(
                verb, abs(r.span_change_pct))
        # Both numbers, because they answer different questions: the first is what
        # the account is already carrying, the second what it would carry if this
        # pair were added. A monitor that shows only one of them is ambiguous
        # exactly when it matters.
        lines.append(
            "{pair} spread trade\n"
            "  Current margin utilization: {now:.0%}  ->  {then:.0%} if added\n"
            "  Maintenance buffer: {state}{chg}\n"
            "  Signal: {action} ({reason})".format(
                pair=r.pair, now=r.margin_utilisation,
                then=r.utilisation_if_added, chg=chg,
                state=r.buffer_state, action=r.action, reason=r.reason))
    return lines


def summary(bd: pd.DataFrame) -> dict:
    if bd is None or bd.empty:
        return {}
    # The board may hold nothing but unpriceable pairs, in which case there is no
    # buffer state to report and saying "nan" would be worse than saying so.
    states = bd["buffer_state"].dropna()
    worst = (min(states, key=lambda s: _BUFFER_ORDER.get(s, 9))
             if len(states) else "none priceable")
    return {
        "pairs": int(len(bd)),
        "blocked": int((bd["action"] == "do not add exposure").sum()),
        "reduced": int((bd["action"] == "add reduced size").sum()),
        "addable": int((bd["action"] == "may add").sum()),
        "unavailable": int((bd["action"] == "unavailable").sum()),
        "max_utilisation_if_added": float(bd["utilisation_if_added"].max()),
        "worst_buffer": worst,
        "max_span_change_pct": float(bd["span_change_pct"].max())
        if bd["span_change_pct"].notna().any() else float("nan"),
    }
