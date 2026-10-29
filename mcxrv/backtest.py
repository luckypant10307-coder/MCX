"""
Walk-forward backtest for two-leg relative-value spreads.

Rules this engine enforces, each of which corresponds to a way these backtests
usually lie:

  no look-ahead        Signals arrive already computed from trailing data only
                       (see `signal`). Entry uses the signal available AT t and
                       fills at t's settlement, adjusted adversely. Nothing reads
                       t+1.
  real contracts       P&L is computed from the settlement prices of the specific
                       contract_ids held, via each one's own quote multiplier.
                       There is no continuous or stitched series anywhere in the
                       P&L path, so no roll-splice artefact can leak in.
  integer lots         Fine-gram neutrality is approximated with whole lots, and
                       the leftover gold exposure is RECORDED, not assumed away.
                       GOLDM carries 99.5 g fine per lot against GOLDGUINEA's
                       7.992 g -- a ratio of 12.45, so a residual always exists.
  four crossings       Opening and closing two legs costs four crossings. Charged
                       explicitly through `costs.leg_cost_inr`.
  calendar discipline  A position is force-closed before the tender window of its
                       nearest-expiring leg. Nothing is ever held through expiry,
                       and no fill is booked on an untradeable settlement.
  honest exits         Every exit carries a reason, and illiquid exits are flagged
                       and charged a widened spread rather than quietly filled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import costs as costs_mod
from . import margin as margin_mod
from . import specs


@dataclass
class BacktestConfig:
    target_notional_per_leg: float = 500_000.0
    # Denominator for return scaling ONLY. Sharpe is scale-invariant so it does not
    # move with this, but annualised return does -- which is why return on actual
    # SPAN margin is reported beside it and is the figure that means something.
    capital_base: float = 2_000_000.0
    max_concurrent: int = 3

    # --- margin ---------------------------------------------------------------
    # Capital blocked is SPAN + ELM on the open book, recomputed daily. A spread is
    # margin-efficient, so the same rupee loss sits on a much smaller base than the
    # notional suggests: accounting for it properly makes a losing spread strategy
    # look worse, not better.
    track_margin: bool = True
    margin_limit: float | None = None   # None: margin never blocks an entry
    span_params: object | None = None   # margin.SpanParameters; None uses defaults
    # Per-session scan ranges from trailing volatility, so required margin RISES
    # when the market gets disturbed -- which is when a losing position is least
    # able to fund it. A single static scan range hides that and understates peak
    # capital. Built by margin.volatility_schedule().
    margin_schedule: dict | None = None

    # --- account and margin calls ---------------------------------------------
    # Indian clearing settles MTM in CASH daily, so the binding constraint on
    # holding a position is free cash, not notional. A position that cannot fund its
    # pay-in is liquidated by the broker at whatever price exists, regardless of how
    # attractive the signal still looks. A backtest that never models that is
    # assuming an infinite balance sheet.
    account_capital: float = 1_000_000.0
    maintenance_pct: float = 1.0        # 100% of requirement: Indian practice
    margin_call_grace_sessions: int = 1  # T+1 pay-in
    force_liquidate_on_call: bool = True
    emit_position_daily: bool = True
    max_holding_sessions: int = 40
    exit_z: float = 0.5
    stop_z: float = 4.5                   # dislocation widened beyond plausibility
    min_sessions_to_expiry_entry: int = 15
    min_sessions_to_expiry_exit: int = 4  # force out at or below this
    # Convergence must be able to HAPPEN inside the contract's remaining life.
    # Entering with 8 sessions left on a 12-session half-life guarantees a forced
    # calendar exit at whatever the spread happens to be, which is a coin flip
    # dressed as a strategy.
    halflife_coverage: float = 2.0         # need >= this many half-lives of runway
    # Maximum tolerated unhedged fine-gram fraction. Above this the position is a
    # directional gold bet, not a spread, so the trade is declined.
    max_residual_frac: float = 0.05
    illiquid_exit_spread_mult: float = 3.0
    allow_pairs_with_goldm: bool = True
    one_position_per_pair: bool = True


@dataclass
class Position:
    pair: str
    sym_a: str
    sym_b: str
    cid_a: str
    cid_b: str
    dir_pair: int            # +1: long a / short b. -1: the reverse.
    lots_a: int
    lots_b: int
    entry_date: object
    entry_close_a: float
    entry_close_b: float
    entry_spread_bp: float
    entry_z: float
    fine_residual_g: float
    residual_frac: float
    entry_cost_inr: float
    notional_a: float
    notional_b: float
    entry_margin: float = 0.0
    # Set when the account could not fund the book. The broker closes the position
    # on the next session at whatever price exists -- the signal gets no say.
    margin_called: bool = False
    sessions_held: int = 0
    mtm_history: list = field(default_factory=list)
    # A STABLE identity, assigned from a per-run counter.
    #
    # This was `id(p)`, which is a memory ADDRESS: CPython reuses it the moment a
    # closed position is freed, so two different trades in one run could share a
    # trade_id. Downstream that merged their daily records into one ledger group
    # (sessions_held became the sum of both) and then fanned the join back out into
    # duplicate rows. Intermittent, silent, and dependent on garbage-collection
    # timing -- so it appeared in a full pipeline run and not in a smaller one.
    trade_id: int = 0


def _multiplier(symbol: str) -> float:
    """Rupees of position value per unit of quote, per lot."""
    sp = specs.get(symbol)
    return sp.trading_unit_g / sp.quote_base_g


def _book_positions(open_pos, lut, t):
    """
    The open book as margin positions, marked at today's settlements.

    Margin is a property of the WHOLE portfolio, not of each trade separately: SPAN
    nets offsetting deltas and grants inter-commodity credits across everything held.
    Summing per-position margins would overstate the requirement, sometimes by a
    large multiple, which is why the book is assembled and priced as one.
    """
    out = []
    for p in open_pos:
        for cid, sym, lots, sign in ((p.cid_a, p.sym_a, p.lots_a, p.dir_pair),
                                     (p.cid_b, p.sym_b, p.lots_b, -p.dir_pair)):
            rec = lut.get((t, cid))
            price = rec.close if rec is not None else (
                p.entry_close_a if cid == p.cid_a else p.entry_close_b)
            s2e = int(rec.sessions_to_expiry) if rec is not None else 99
            sig = getattr(rec, "sigma_daily", None) if rec is not None else None
            oi = getattr(rec, "open_interest", None) if rec is not None else None
            out.append(margin_mod.MarginPosition(
                symbol=sym, contract_id=cid, lots=int(sign) * int(lots),
                price=float(price), sessions_to_expiry=s2e,
                sigma_daily=float(sig) if sig is not None and np.isfinite(sig) else None,
                open_interest=float(oi) if oi is not None and np.isfinite(oi) else None))
    return out


def _size_legs(sym_a: str, sym_b: str, close_a: float, close_b: float,
               max_lots_a: float, max_lots_b: float,
               cfg: BacktestConfig):
    """
    Choose whole-lot sizes that achieve fine-gram neutrality within tolerance.

    Returns (lots_a, lots_b, fine_residual_g, residual_frac), or None when no
    integer combination inside the participation caps can hedge to tolerance.

    Returning None matters. Lot sizes do not divide: GOLDM carries 99.5 g fine per
    lot against GOLDGUINEA's 7.992 g, and in lowest terms that ratio is 24875/1998 --
    so the smallest EXACTLY neutral combination is 1998 GOLDM lots against 24875
    GOLDGUINEA, about Rs 24 crore a leg. Exact neutrality is therefore unattainable
    in practice and some residual gold exposure always exists; the job is to keep it
    below tolerance, measure it, and report it.

    When even the tolerance cannot be met -- typically because a participation cap on
    a thin leg leaves too few lots to hedge against the other leg's granularity --
    forcing the trade anyway leaves a large unhedged gold position, and the resulting
    P&L is a directional gold bet mislabelled as relative value. The honest response
    is to decline, and to let the diagnostic count say how often granularity binds.

    Search: for each feasible lots_a, take the nearest integer lots_b, keep the
    combinations that hedge to tolerance, and prefer the one whose notional lands
    closest to target (ties to the larger size).
    """
    sp_a, sp_b = specs.get(sym_a), specs.get(sym_b)
    mult_a, mult_b = _multiplier(sym_a), _multiplier(sym_b)

    lot_val_a = close_a * mult_a
    lot_val_b = close_b * mult_b
    if lot_val_a <= 0 or lot_val_b <= 0:
        return None

    cap_a = int(np.floor(max_lots_a))
    cap_b = int(np.floor(max_lots_b))
    if cap_a < 1 or cap_b < 1:
        return None

    f_a, f_b = sp_a.fine_g_per_lot, sp_b.fine_g_per_lot
    target = cfg.target_notional_per_leg

    best = None
    # Cap the search: beyond the target notional there is no reason to go bigger.
    hi_a = min(cap_a, max(1, int(np.ceil(target / lot_val_a)) * 2))
    for n_a in range(1, hi_a + 1):
        fine_a = n_a * f_a
        n_b = int(round(fine_a / f_b))
        if n_b < 1 or n_b > cap_b:
            continue
        resid = fine_a - n_b * f_b
        frac = abs(resid) / max(fine_a, 1e-9)
        if frac > cfg.max_residual_frac:
            continue
        notional = n_a * lot_val_a
        score = (abs(notional - target), -n_a)
        if best is None or score < best[0]:
            best = (score, n_a, n_b, resid, frac)

    if best is None:
        return None
    _score, n_a, n_b, resid, frac = best
    return n_a, n_b, resid, frac


def run(rows: pd.DataFrame, pair_sig: pd.DataFrame, cfg: BacktestConfig | None = None,
        cost_cfg: costs_mod.CostConfig | None = None):
    """
    Execute the walk-forward backtest.

    `rows` must be the liquidity-annotated frame (from `liquidity.annotate`).
    `pair_sig` must be `signal.pair_spreads` output.

    Returns (trades, daily, diag):
      trades  one row per closed position, with gross and net P&L and exit reason
      daily   per-session equity, mark-to-market, costs, open position count
      diag    counts of why entries were skipped -- the audit trail for silence
    """
    cfg = cfg or BacktestConfig()
    cost_cfg = cost_cfg or costs_mod.CostConfig()

    # Fast lookup of the executable row for a contract on a date.
    rows = rows.copy()
    # Trailing daily volatility per contract, for the delivery-period VaR. Strictly
    # backward-looking: shift(1) so today's own return never informs today's margin.
    rows = rows.sort_values(["contract_id", "date"])
    _lr = np.log(rows["norm_close"].astype(float)).groupby(
        rows["contract_id"], sort=False).diff()
    rows["sigma_daily"] = (_lr.groupby(rows["contract_id"], sort=False)
                           .transform(lambda x: x.ewm(halflife=30, min_periods=15)
                                      .std().shift(1)))
    lut = {}
    for rec in rows[["date", "contract_id", "close", "volume", "max_lots",
                     "half_spread_bp", "tradable", "sessions_to_expiry",
                     "in_tender", "norm_close", "sigma_daily",
                     "open_interest"]].itertuples(index=False):
        lut[(rec.date, rec.contract_id)] = rec

    # Tradable contracts per (date, symbol), ranked by volume descending.
    #
    # A ranked list rather than a single front-month pick, because the front month
    # is liquid but short-lived: in its liquid window it has only ~21 sessions,
    # which cannot accommodate a 12-session half-life needing two half-lives of
    # runway. A real trader rolls out to the next expiry in that situation, and the
    # engine has to be allowed the same choice or the runway gate silently vetoes
    # every trade and the result looks like "no signal" when it is "wrong contract".
    trad = rows[rows["tradable"]]
    cands: dict = {}
    if not trad.empty:
        cols = ["date", "symbol", "contract_id", "volume", "sessions_to_expiry"]
        for rec in (trad[cols].sort_values(["date", "symbol", "volume"],
                                          ascending=[True, True, False])
                    .itertuples(index=False)):
            cands.setdefault((rec.date, rec.symbol), []).append(
                (rec.contract_id, int(rec.sessions_to_expiry)))

    def pick_contract(day, symbol, need_sessions):
        """Most-liquid tradable contract with enough remaining runway."""
        for cid, s2e in cands.get((day, symbol), ()):
            if s2e >= need_sessions:
                return cid
        return None

    sig_by_date = {d: g for d, g in pair_sig.groupby("date")}
    dates = sorted(rows["date"].unique())

    span_params = cfg.span_params or margin_mod.SpanParameters()

    open_pos: list = []
    trades, daily_rows = [], []
    diag = {}
    realised = 0.0
    peak_margin = 0.0
    pos_daily: list = []
    call_streak = 0
    forced_liquidations: list = []
    next_trade_id = 0

    def bump(k, n=1):
        diag[k] = diag.get(k, 0) + n

    for t in dates:
        day_cost = 0.0
        day_realised = 0.0

        # ---------------- manage open positions first ----------------
        still_open = []
        for p in open_pos:
            ra = lut.get((t, p.cid_a))
            rb = lut.get((t, p.cid_b))
            if ra is None or rb is None:
                # Contract vanished from the tape: carry at last mark, try again.
                p.mtm_history.append(p.mtm_history[-1] if p.mtm_history else 0.0)
                still_open.append(p)
                bump("stale_mark_no_quote")
                continue

            p.sessions_held += 1
            mult_a, mult_b = _multiplier(p.sym_a), _multiplier(p.sym_b)
            pnl_a = p.dir_pair * p.lots_a * (ra.close - p.entry_close_a) * mult_a
            pnl_b = -p.dir_pair * p.lots_b * (rb.close - p.entry_close_b) * mult_b
            gross = pnl_a + pnl_b
            p.mtm_history.append(gross)

            # current signal state for this pair
            g = sig_by_date.get(t)
            z_now = np.nan
            spread_now = np.nan
            if g is not None:
                gg = g[g["pair"] == p.pair]
                if len(gg):
                    z_now = float(gg["z"].iloc[0])
                    spread_now = float(gg["spread_bp"].iloc[0])

            # ---- exit decision, in priority order ----
            reason = None
            min_sessions = min(ra.sessions_to_expiry, rb.sessions_to_expiry)
            if p.margin_called:
                # Outranks everything: the position is gone whatever the signal says.
                reason = "margin_call"
            elif min_sessions <= cfg.min_sessions_to_expiry_exit or ra.in_tender or rb.in_tender:
                reason = "calendar"
            elif p.sessions_held >= cfg.max_holding_sessions:
                reason = "max_holding"
            elif np.isfinite(z_now) and abs(z_now) <= cfg.exit_z:
                reason = "converged"
            elif np.isfinite(z_now) and abs(z_now) >= cfg.stop_z:
                reason = "stop"

            if reason is None:
                still_open.append(p)
                continue

            # ---- close both legs ----
            illiquid = (not ra.tradable) or (not rb.tradable) or p.margin_called
            hs_a = ra.half_spread_bp * (cfg.illiquid_exit_spread_mult if illiquid else 1.0)
            hs_b = rb.half_spread_bp * (cfg.illiquid_exit_spread_mult if illiquid else 1.0)
            notional_a = abs(ra.close * mult_a * p.lots_a)
            notional_b = abs(rb.close * mult_b * p.lots_b)
            side_a = "sell" if p.dir_pair > 0 else "buy"
            side_b = "buy" if p.dir_pair > 0 else "sell"
            c_a = costs_mod.leg_cost_inr(notional_a, side_a, hs_a, p.lots_a,
                                         max(ra.max_lots, 1), cost_cfg)
            c_b = costs_mod.leg_cost_inr(notional_b, side_b, hs_b, p.lots_b,
                                         max(rb.max_lots, 1), cost_cfg)
            exit_cost = c_a["total_inr"] + c_b["total_inr"]

            net = gross - p.entry_cost_inr - exit_cost
            day_cost += exit_cost
            day_realised += net
            realised += net

            trades.append({
                "pair": p.pair, "sym_a": p.sym_a, "sym_b": p.sym_b,
                "cid_a": p.cid_a, "cid_b": p.cid_b, "dir_pair": p.dir_pair,
                "lots_a": p.lots_a, "lots_b": p.lots_b,
                "entry_date": p.entry_date, "exit_date": t,
                "sessions_held": p.sessions_held,
                "entry_z": p.entry_z, "exit_z": z_now,
                "entry_spread_bp": p.entry_spread_bp, "exit_spread_bp": spread_now,
                "entry_close_a": p.entry_close_a, "exit_close_a": ra.close,
                "entry_close_b": p.entry_close_b, "exit_close_b": rb.close,
                "notional_a": p.notional_a, "notional_b": p.notional_b,
                "fine_residual_g": p.fine_residual_g,
                "residual_frac": p.residual_frac,
                "gross_pnl": gross, "entry_cost": p.entry_cost_inr,
                "exit_cost": exit_cost, "total_cost": p.entry_cost_inr + exit_cost,
                "net_pnl": net, "exit_reason": reason,
                "illiquid_exit": bool(illiquid),
                "margin_called": bool(p.margin_called),
                "trade_id": p.trade_id,
                "gross_bp": gross / max(p.notional_a, 1e-9) * 10000.0,
                "net_bp": net / max(p.notional_a, 1e-9) * 10000.0,
                "entry_margin": p.entry_margin,
                "return_on_margin": (net / p.entry_margin
                                     if p.entry_margin > 0 else np.nan),
            })
        open_pos = still_open

        # ---------------- consider new entries ----------------
        g = sig_by_date.get(t)
        if g is not None and len(open_pos) < cfg.max_concurrent:
            cand = g[g["signal_ok"] & (g["direction"] != 0)]
            if not cfg.allow_pairs_with_goldm:
                cand = cand[cand["maturity_matched"]]
            # strongest dislocation first
            cand = cand.reindex(cand["z"].abs().sort_values(ascending=False).index)

            held_pairs = {p.pair for p in open_pos}
            for rec in cand.itertuples(index=False):
                if len(open_pos) >= cfg.max_concurrent:
                    break
                if cfg.one_position_per_pair and rec.pair in held_pairs:
                    bump("skip_pair_already_open")
                    continue

                # Runway the signal needs: enough half-lives to converge, plus the
                # buffer we insist on keeping before the tender window.
                hl = getattr(rec, "halflife", np.nan)
                need = cfg.min_sessions_to_expiry_entry
                if np.isfinite(hl):
                    need = max(need, int(np.ceil(cfg.halflife_coverage * hl))
                               + cfg.min_sessions_to_expiry_exit)

                cid_a = pick_contract(t, rec.sym_a, need)
                cid_b = pick_contract(t, rec.sym_b, need)
                if cid_a is None or cid_b is None:
                    # Distinguish "nothing tradable at all" from "nothing with
                    # enough runway": they are different problems with different
                    # fixes, and collapsing them hides which one binds.
                    if not cands.get((t, rec.sym_a)) or not cands.get((t, rec.sym_b)):
                        bump("skip_no_tradable_contract")
                    else:
                        bump("skip_no_contract_with_runway")
                    continue
                ra, rb = lut[(t, cid_a)], lut[(t, cid_b)]

                sized = _size_legs(rec.sym_a, rec.sym_b, ra.close, rb.close,
                                   ra.max_lots, rb.max_lots, cfg)
                if sized is None:
                    bump("skip_cannot_size")
                    continue
                lots_a, lots_b, resid_g, resid_frac = sized

                # Margin is the real constraint on how many positions can be held.
                # Check the WHOLE prospective book, not this trade alone, because
                # SPAN nets across everything open.
                if cfg.track_margin and cfg.margin_limit is not None:
                    trial = margin_mod.MarginPosition
                    prospective = _book_positions(open_pos, lut, t) + [
                        trial(rec.sym_a, cid_a, int(rec.direction) * lots_a,
                              float(ra.close), int(ra.sessions_to_expiry)),
                        trial(rec.sym_b, cid_b, -int(rec.direction) * lots_b,
                              float(rb.close), int(rb.sessions_to_expiry)),
                    ]
                    need = margin_mod.initial_margin(
                        prospective,
                        margin_mod.params_for_session(span_params, t,
                                                      cfg.margin_schedule))
                    if need > cfg.margin_limit:
                        bump("skip_margin_limit")
                        continue

                mult_a, mult_b = _multiplier(rec.sym_a), _multiplier(rec.sym_b)
                notional_a = abs(ra.close * mult_a * lots_a)
                notional_b = abs(rb.close * mult_b * lots_b)

                # direction: +1 means spread too low -> long a, short b
                side_a = "buy" if rec.direction > 0 else "sell"
                side_b = "sell" if rec.direction > 0 else "buy"
                c_a = costs_mod.leg_cost_inr(notional_a, side_a, ra.half_spread_bp,
                                             lots_a, max(ra.max_lots, 1), cost_cfg)
                c_b = costs_mod.leg_cost_inr(notional_b, side_b, rb.half_spread_bp,
                                             lots_b, max(rb.max_lots, 1), cost_cfg)
                entry_cost = c_a["total_inr"] + c_b["total_inr"]

                next_trade_id += 1
                open_pos.append(Position(
                    pair=rec.pair, sym_a=rec.sym_a, sym_b=rec.sym_b,
                    cid_a=cid_a, cid_b=cid_b, dir_pair=int(rec.direction),
                    lots_a=lots_a, lots_b=lots_b, entry_date=t,
                    entry_close_a=ra.close, entry_close_b=rb.close,
                    entry_spread_bp=float(rec.spread_bp), entry_z=float(rec.z),
                    fine_residual_g=resid_g, residual_frac=resid_frac,
                    entry_cost_inr=entry_cost,
                    notional_a=notional_a, notional_b=notional_b,
                    entry_margin=margin_mod.initial_margin([
                        margin_mod.MarginPosition(
                            rec.sym_a, cid_a, int(rec.direction) * lots_a,
                            float(ra.close), int(ra.sessions_to_expiry)),
                        margin_mod.MarginPosition(
                            rec.sym_b, cid_b, -int(rec.direction) * lots_b,
                            float(rb.close), int(rb.sessions_to_expiry)),
                    ], span_params) if cfg.track_margin else 0.0,
                    trade_id=next_trade_id,
                ))
                held_pairs.add(rec.pair)
                day_cost += entry_cost
                bump("entries")

        unreal = sum(p.mtm_history[-1] - p.entry_cost_inr
                     for p in open_pos if p.mtm_history)

        # --- margin on the book as it stands at the close ---
        day_margin = 0.0
        day_span = 0.0
        day_credit = 0.0
        if cfg.track_margin and open_pos:
            sp_t = margin_mod.params_for_session(span_params, t, cfg.margin_schedule)
            m = margin_mod.span_margin(_book_positions(open_pos, lut, t), sp_t)
            day_margin = m["initial_margin"]
            day_span = m["span"]
            day_credit = m["inter_credit"]

            # Per-position record. Margin is attributed MARGINALLY -- the book with
            # the position minus the book without it -- because SPAN nets and credits
            # across the whole portfolio, so a standalone figure would double-count
            # risk that the book already offsets.
            if cfg.emit_position_daily:
                for p in open_pos:
                    solo = margin_mod.span_margin(
                        _book_positions([p], lut, t), sp_t)
                    others = [q for q in open_pos if q is not p]
                    without = (margin_mod.span_margin(
                        _book_positions(others, lut, t), sp_t)["initial_margin"]
                        if others else 0.0)
                    pos_daily.append({
                        "trade_id": p.trade_id, "pair": p.pair, "date": t,
                        "sessions_held": p.sessions_held,
                        "notional": solo["notional"],
                        "net_delta": solo["net_delta"],
                        "initial_margin": solo["initial_margin"],
                        "marginal_margin": day_margin - without,
                        "span": solo["span"],
                        "elm": solo["elm"],
                        "additional": solo["additional"],
                        "delivery": solo["delivery"],
                        "concentration": solo["concentration"],
                        "inter_credit": solo["inter_credit"],
                        "floor_binds": solo["floor_binds"],
                        "mtm_pnl": (p.mtm_history[-1] - p.entry_cost_inr
                                    if p.mtm_history else -p.entry_cost_inr),
                        "margin_call": False, "forced_exit": False,
                    })
        peak_margin = max(peak_margin, day_margin)

        # --- account solvency: can the book actually be funded? ---
        equity_now = cfg.account_capital + realised + unreal
        required = day_margin * cfg.maintenance_pct
        in_call = bool(open_pos) and equity_now < required
        if in_call:
            call_streak += 1
            bump("margin_call_sessions")
            for rec in pos_daily[-len(open_pos):]:
                rec["margin_call"] = True
        else:
            call_streak = 0

        if (in_call and cfg.force_liquidate_on_call
                and call_streak > cfg.margin_call_grace_sessions and open_pos):
            # Liquidate the worst position first, which is what a broker does.
            victim = min(open_pos, key=lambda q: (q.mtm_history[-1]
                                                  if q.mtm_history else 0.0))
            forced_liquidations.append((t, victim.pair))
            victim.margin_called = True
            bump("forced_exit_margin_call")
            call_streak = 0

        daily_rows.append({
            "date": t, "realised_cum": realised, "unrealised": unreal,
            "equity": realised + unreal, "day_realised": day_realised,
            "day_cost": day_cost, "n_open": len(open_pos),
            "margin": day_margin, "span": day_span, "inter_credit": day_credit,
        })

    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df["d_equity"] = daily_df["equity"].diff().fillna(
            daily_df["equity"].iloc[0])
        daily_df["ret"] = daily_df["d_equity"] / cfg.capital_base
    diag_df = (pd.DataFrame(sorted(diag.items()), columns=["reason", "count"])
               if diag else pd.DataFrame(columns=["reason", "count"]))
    if not daily_df.empty:
        daily_df.attrs["peak_margin"] = peak_margin
    pos_daily_df = (pd.DataFrame(pos_daily) if pos_daily
                    else pd.DataFrame())
    if not pos_daily_df.empty and forced_liquidations:
        forced = {(d, pr) for d, pr in forced_liquidations}
        pos_daily_df["forced_exit"] = [
            (r.date, r.pair) in forced
            for r in pos_daily_df.itertuples(index=False)]
    return trades_df, daily_df, diag_df, pos_daily_df


def performance(trades: pd.DataFrame, daily: pd.DataFrame,
                cfg: BacktestConfig | None = None) -> dict:
    """
    Headline statistics, gross and net, with the cost drag stated separately.

    Deliberately reports both so the reader can see how much of the gross result
    survives friction -- which for thin contracts is the entire question.
    """
    cfg = cfg or BacktestConfig()
    out = {"n_trades": int(len(trades))}
    if trades.empty:
        out.update({"note": "no trades taken"})
        return out

    out["gross_pnl"] = float(trades["gross_pnl"].sum())
    out["total_cost"] = float(trades["total_cost"].sum())
    out["net_pnl"] = float(trades["net_pnl"].sum())
    out["cost_share_of_gross"] = (
        float(trades["total_cost"].sum() / abs(trades["gross_pnl"].sum()))
        if trades["gross_pnl"].sum() != 0 else np.inf)
    out["hit_rate_gross"] = float((trades["gross_pnl"] > 0).mean())
    out["hit_rate_net"] = float((trades["net_pnl"] > 0).mean())
    out["mean_gross_bp"] = float(trades["gross_bp"].mean())
    out["mean_net_bp"] = float(trades["net_bp"].mean())
    out["median_net_bp"] = float(trades["net_bp"].median())
    out["mean_sessions_held"] = float(trades["sessions_held"].mean())
    out["mean_residual_frac"] = float(trades["residual_frac"].mean())
    out["max_residual_frac"] = float(trades["residual_frac"].max())
    out["illiquid_exit_rate"] = float(trades["illiquid_exit"].mean())
    out["exit_reasons"] = trades["exit_reason"].value_counts().to_dict()

    # --- capital actually blocked ---
    if "entry_margin" in trades.columns and trades["entry_margin"].max() > 0:
        peak = float(daily.attrs.get("peak_margin", 0.0)) if not daily.empty else 0.0
        if peak <= 0 and not daily.empty and "margin" in daily.columns:
            peak = float(daily["margin"].max())
        out["peak_margin"] = peak
        out["mean_entry_margin"] = float(trades["entry_margin"].mean())
        out["margin_pct_notional"] = float(
            (trades["entry_margin"] / (trades["notional_a"] + trades["notional_b"]))
            .mean())
        if peak > 0:
            out["return_on_peak_margin"] = float(trades["net_pnl"].sum() / peak)
            if not daily.empty:
                yrs = max(len(daily), 1) / 252.0
                out["ann_return_on_margin"] = out["return_on_peak_margin"] / yrs
        if "return_on_margin" in trades.columns:
            out["mean_return_on_margin_per_trade"] = float(
                trades["return_on_margin"].mean())

    if not daily.empty and daily["ret"].std() > 0:
        r = daily["ret"].dropna()
        out["ann_return"] = float(r.mean() * 252)
        out["ann_vol"] = float(r.std() * np.sqrt(252))
        out["sharpe"] = float(r.mean() / r.std() * np.sqrt(252))
        eq = daily["equity"]
        out["max_drawdown_inr"] = float((eq - eq.cummax()).min())
        out["n_sessions"] = int(len(daily))

    # A paired t-test on per-trade net bp. Trades overlap in time, so this is
    # indicative rather than a clean significance claim -- labelled as such.
    x = trades["net_bp"].to_numpy(dtype=float)
    if len(x) > 2 and np.std(x, ddof=1) > 0:
        out["net_bp_tstat_naive"] = float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))
        out["tstat_caveat"] = ("overlapping holding periods; not an independent "
                              "sample, treat as indicative only")
    return out
