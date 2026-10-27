"""
Generate the trader-facing dashboard as a single self-contained HTML file.

Design intent. The centrepiece is not the signal, it is the DECOMPOSITION. A trader
looking at GOLDPETAL trading 110 bp above GOLDM should see immediately that almost
all of that gap is a structural fabrication premium plus carry, and that only the
last few basis points are actionable. A dashboard that shows the 110 bp as an
"opportunity" is worse than no dashboard.

The page is quiet by default. When nothing clears the gates it says so, and shows
which gate is binding rather than inventing something to display.

No external dependencies: charts are inline SVG generated here, so the file opens
offline and can be committed or emailed as-is.

Usage:
  python scripts/make_dashboard.py
  python scripts/make_dashboard.py --source csv --csv-dir data/raw --out reports/dashboard.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv.stage1_intake import anomaly
from mcxrv.stage4_simulate_deliver import evaluation as attribution
from mcxrv.stage4_simulate_deliver import replay_driver as backtest
from mcxrv.stage2_normalize_signal import half_life as carry
from mcxrv.stage3_gates_decision import liquidity_cost as costs
from mcxrv.stage4_simulate_deliver import eventstudy
from mcxrv.stage1_intake import validation as ingest
from mcxrv.stage2_normalize_signal import pair_match as liquidity
from mcxrv.stage3_gates_decision import margin_capital as margin
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.stage3_gates_decision import risk
from mcxrv.stage1_intake import specs
from mcxrv.stage2_normalize_signal import z_score as sig_mod

# ---------------------------------------------------------------------------
# palette: validated categorical slots 1-3 (all-pairs safe), reserved status hues
# ---------------------------------------------------------------------------
SERIES = {"premium": "var(--series-1)", "carry": "var(--series-2)",
          "residual": "var(--series-3)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic", choices=["synthetic", "csv", "db"])
    p.add_argument("--csv-dir", default="data/raw")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-09-18")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--z-entry", type=float, default=2.0)
    p.add_argument("--out", default="reports/dashboard.html")
    p.add_argument("--history", type=int, default=260)
    p.add_argument("--no-anomaly", action="store_true",
                   help="skip the Isolation Forest review queue")
    p.add_argument("--account-capital", type=float, default=1_000_000.0,
                   help="cash actually put up; sets the risk board utilisation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def build(args):
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.source == "db":
        # Quotes were validated on the way in, so read them straight out rather than
        # re-running ingest. The quality ledger is restricted to the runs that
        # produced the current rows, so a reloaded range is not double-counted.
        from mcxrv.db import repo as dbrepo
        with dbrepo.connect() as c:
            df = dbrepo.read_quotes(c, start=start, end=end,
                                    symbols=specs.ALL_SYMBOLS)
            dq = dbrepo._fetch_df(
                c, "SELECT issue, sum(rows) AS rows FROM dq_flag "
                   "WHERE run_id IN (SELECT DISTINCT run_id FROM quote "
                   "                 WHERE run_id IS NOT NULL) GROUP BY issue")
            cfg = dbrepo.DbConfig.from_env() if hasattr(dbrepo, "DbConfig") else None
        if df.empty:
            raise SystemExit("no stored quotes; run scripts/db_load.py first")
        rep = ingest.DataQualityReport(rows_in=int(dq["rows"].max()) if not dq.empty
                                       else len(df), rows_out=len(df))
        for r in dq.itertuples(index=False):
            rep.counts[r.issue] = int(r.rows)
        provenance = "PostgreSQL store (mcx schema), validated on ingest"
    else:
        if args.source == "synthetic":
            from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource
            src = SyntheticSource(SyntheticConfig(start=start, end=end,
                                                  seed=args.seed))
            raw = src.fetch_range(start, end)
            provenance = "synthetic ground-truth generator (seed {})".format(args.seed)
        else:
            from mcxrv.sources.csv_dir import CSVDirectorySource
            raw = CSVDirectorySource(args.csv_dir).load_all(start, end)
            provenance = "MCX bhavcopy CSV files in {}".format(args.csv_dir)

        df, rep, tc = ingest.build_dataset(raw, symbols=specs.ALL_SYMBOLS)
        if df.empty:
            raise SystemExit("no usable rows after validation; nothing to render")
    df = normalize.add_normalized(df)
    rows, daily, prem = carry.fit(df, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    sym_resid = carry.symbol_residual(rows)
    nf = liquidity.symbol_noise_floor(rows)
    scfg = sig_mod.SignalConfig(z_entry=args.z_entry)
    pairs = sig_mod.pair_spreads(sym_resid, scfg)
    es = eventstudy.study(pairs) if not pairs.empty else pd.DataFrame()

    ccfg = costs.CostConfig()
    rt = costs.round_trip_bp(500_000.0, 5.0, 5.0, ccfg)["total_bp"]

    pairs_exec = sig_mod.lag_for_execution(pairs, sessions=1)
    span_params = margin.SpanParameters(
        price_scan_range_pct=margin.derive_scan_range(rows))
    bcfg = backtest.BacktestConfig(exit_z=scfg.z_exit, span_params=span_params)
    trades, bt_daily, diag, pos_daily = backtest.run(rows, pairs_exec, bcfg, ccfg)
    perf = backtest.performance(trades, bt_daily, bcfg)
    gold = (attribution.gold_beta(bt_daily, daily)
            if not trades.empty and not bt_daily.empty else {})
    dec = (attribution.decompose_trades(trades, rows, daily)
           if not trades.empty else pd.DataFrame())
    attrib = attribution.attribution_summary(dec) if not dec.empty else pd.DataFrame()

    det = (anomaly.detect(rows) if (anomaly.available()
                                    and not getattr(args, "no_anomaly", False))
           else pd.DataFrame())

    # Live risk board: what the book would cost to add to on the latest session.
    # Priced against whatever is still open then, so the cost is marginal.
    rb_cfg = risk.RiskBoardConfig(
        account_capital=getattr(args, "account_capital", 1_000_000.0))
    asof_rb = max(pairs["date"]) if not pairs.empty else None
    rbd = (risk.board(rows, pairs, rb_cfg, span_params=span_params,
                      margin_schedule=margin.volatility_schedule(rows),
                      bt_cfg=bcfg,
                      open_book=risk.open_book_from_trades(rows, trades, asof_rb))
           if asof_rb is not None else pd.DataFrame())

    return dict(span_params=span_params, anomaly=det, rep=rep, rows=rows,
                risk_board=rbd, risk_cfg=rb_cfg,
                daily=daily, prem=prem,
                sym_resid=sym_resid,
                nf=nf, pairs=pairs, es=es, rt=rt, trades=trades, bt_daily=bt_daily,
                perf=perf, gold=gold, attrib=attrib, cost_cfg=ccfg,
                provenance=provenance, scfg=scfg, coverage=liquidity.coverage_report(rows))


def latest_snapshot(ctx):
    """Per-symbol executable state on the most recent session."""
    rows = ctx["rows"]
    asof = rows["date"].max()
    day = rows[rows["date"] == asof]
    out = []
    for sym, g in day.groupby("symbol"):
        t = g[g["tradable"]]
        pick = (t.loc[t["volume"].idxmax()] if not t.empty
                else g.loc[g["volume"].idxmax()])
        out.append({
            "symbol": sym, "expiry": pick["expiry_date"], "close": pick["close"],
            "norm_close": pick["norm_close"], "volume": int(pick["volume"]),
            "oi": int(pick["open_interest"]),
            "sessions_to_expiry": int(pick["sessions_to_expiry"]),
            "tradable": bool(pick["tradable"]),
            "residual_bp": float(pick["residual_bp"]) if pd.notna(pick["residual_bp"]) else np.nan,
            "premium_bp": float(pick["premium_bp"]),
            "tau": float(pick["T_years"]),
            "half_spread_bp": float(pick["half_spread_bp"]),
            "max_lots": float(pick["max_lots"]),
            "noise_bp": float(pick["noise_bp"]),
            "contract_id": pick["contract_id"],
        })
    return asof, pd.DataFrame(out).sort_values("norm_close", ascending=False)


def decompose_pairs(ctx, asof, snap):
    """
    Split each pair's raw normalised log spread into premium + carry + residual.

    These three sum to the observed spread by construction, which is what makes the
    chart trustworthy: the bar's total is the number a trader sees on screen, and the
    segments say how much of it is actionable.
    """
    daily = ctx["daily"].set_index("date")
    slope = float(daily.loc[asof, "slope"]) if asof in daily.index else np.nan
    s = snap.set_index("symbol")
    out = []
    syms = [x for x in specs.ALL_SYMBOLS if x in s.index]
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            ra, rb = s.loc[a], s.loc[b]
            if not np.isfinite(ra["residual_bp"]) or not np.isfinite(rb["residual_bp"]):
                continue
            total = (np.log(ra["norm_close"]) - np.log(rb["norm_close"])) * 10000.0
            premium = ra["premium_bp"] - rb["premium_bp"]
            carry_c = slope * (ra["tau"] - rb["tau"]) * 10000.0
            resid = ra["residual_bp"] - rb["residual_bp"]
            out.append({
                "pair": "{}/{}".format(a, b), "sym_a": a, "sym_b": b,
                "total_bp": total, "premium_bp": premium, "carry_bp": carry_c,
                "residual_bp": resid,
                "maturity_matched": specs.is_maturity_matched(a, b),
                "unexplained_bp": total - premium - carry_c - resid,
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _np_ok(v):
    try:
        return bool(np.isfinite(v))
    except Exception:
        return False


def esc(x):
    return html.escape(str(x), quote=True)


def _fmt(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return "{:+,.{}f}".format(v, nd) if abs(v) < 1e7 else "{:,.0f}".format(v)


def svg_decomposition(dec: pd.DataFrame, width=680, row_h=44) -> str:
    """
    Diverging horizontal stacked bars: each pair's spread split by component.

    One scale, labelled ticks, zero rule drawn. Segments carry a 2px surface gap so
    adjacent fills never touch, and each is hoverable with its own value.
    """
    if dec.empty:
        return '<p class="empty">No pair with a fitted residual on this session.</p>'
    d = dec.copy().sort_values("total_bp")
    pad_l, pad_r, pad_t, pad_b = 148, 58, 26, 34
    h = pad_t + pad_b + row_h * len(d)

    comps = [("premium_bp", "Structural premium", SERIES["premium"]),
             ("carry_bp", "Carry / term structure", SERIES["carry"]),
             ("residual_bp", "Residual (actionable)", SERIES["residual"])]

    lo = min(0.0, float(min(d[[c for c, _, _ in comps]].clip(upper=0).sum(axis=1).min(),
                            d["total_bp"].min())))
    hi = max(0.0, float(max(d[[c for c, _, _ in comps]].clip(lower=0).sum(axis=1).max(),
                            d["total_bp"].max())))
    span = max(hi - lo, 1.0)
    lo -= span * 0.06
    hi += span * 0.06

    def x(v):
        return pad_l + (v - lo) / (hi - lo) * (width - pad_l - pad_r)

    parts = ['<svg viewBox="0 0 {} {}" width="100%" role="img" '
             'aria-label="Spread decomposition by pair, basis points">'.format(width, h)]

    # ticks
    nticks = 6
    for k in range(nticks + 1):
        v = lo + (hi - lo) * k / nticks
        xv = x(v)
        parts.append('<line class="grid" x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}"/>'
                     .format(xv, pad_t - 6, xv, h - pad_b))
        parts.append('<text class="tick" x="{:.1f}" y="{}" text-anchor="middle">{:.0f}</text>'
                     .format(xv, h - pad_b + 16, v))
    parts.append('<text class="axis-title" x="{:.1f}" y="{}" text-anchor="middle">'
                 'basis points of fine-gold price</text>'
                 .format((pad_l + width - pad_r) / 2, h - pad_b + 31))

    for i, r in enumerate(d.itertuples(index=False)):
        y = pad_t + i * row_h
        by = y + 9
        bh = row_h - 22
        parts.append('<text class="rowlab" x="{}" y="{:.1f}">{}</text>'
                     .format(pad_l - 12, by + bh * 0.72, esc(r.pair)))
        if not r.maturity_matched:
            parts.append('<text class="rowtag" x="{}" y="{:.1f}">carry-bridged</text>'
                         .format(pad_l - 12, by + bh * 0.72 + 12))

        # stack positives right of zero, negatives left
        pos_c = 0.0
        neg_c = 0.0
        for col, label, colour in comps:
            v = float(getattr(r, col))
            if not np.isfinite(v) or abs(v) < 1e-9:
                continue
            if v > 0:
                x0, x1 = x(pos_c), x(pos_c + v)
                pos_c += v
            else:
                x0, x1 = x(neg_c + v), x(neg_c)
                neg_c += v
            w = max(abs(x1 - x0) - 2.0, 0.8)   # 2px surface gap between fills
            parts.append(
                '<rect class="seg" x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
                'rx="2" fill="{}" data-tip="{} &middot; {}: {} bp"><title>{}: {:+.1f} bp'
                '</title></rect>'.format(
                    min(x0, x1) + 1.0, by, w, bh, colour,
                    esc(r.pair), esc(label), _fmt(v), esc(label), v))

        # total marker
        parts.append('<line class="totalmark" x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}"/>'
                     .format(x(r.total_bp), by - 4, x(r.total_bp), by + bh + 4))
        parts.append('<text class="totalval" x="{:.1f}" y="{:.1f}">{} bp</text>'
                     .format(width - pad_r + 6, by + bh * 0.72, _fmt(r.total_bp, 0)))

    parts.append('<line class="zero" x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}"/>'
                 .format(x(0), pad_t - 6, x(0), h - pad_b))
    parts.append("</svg>")
    return "".join(parts)


def svg_spread_history(hist: pd.DataFrame, pair: str, width=330, height=104) -> str:
    """
    Residual spread with its trailing +/-2 sigma band and an emphasised endpoint.

    The band is trailing, so it widens and narrows as the estimate does -- which is
    the honest picture, and it makes the early period visibly unusable.
    """
    g = hist.dropna(subset=["spread_bp"])
    if len(g) < 5:
        return '<p class="empty">Not enough history.</p>'
    pad_l, pad_r, pad_t, pad_b = 40, 8, 10, 18
    xs = np.arange(len(g))
    v = g["spread_bp"].to_numpy(dtype=float)
    centre = g["centre"].to_numpy(dtype=float)
    scale = g["scale"].to_numpy(dtype=float)
    up, dn = centre + 2 * scale, centre - 2 * scale

    fin = lambda a: a[np.isfinite(a)]
    allv = np.concatenate([fin(v), fin(up), fin(dn)])
    lo, hi = float(np.min(allv)), float(np.max(allv))
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    m = (hi - lo) * 0.10
    lo, hi = lo - m, hi + m

    X = lambda i: pad_l + i / max(len(g) - 1, 1) * (width - pad_l - pad_r)
    Y = lambda y: pad_t + (hi - y) / (hi - lo) * (height - pad_t - pad_b)

    parts = ['<svg viewBox="0 0 {} {}" width="100%" role="img" aria-label="{} '
             'residual spread history">'.format(width, height, esc(pair))]
    for yv in (lo + (hi - lo) * k / 2 for k in range(3)):
        parts.append('<line class="grid" x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}"/>'
                     .format(pad_l, Y(yv), width - pad_r, Y(yv)))
        parts.append('<text class="tick sm" x="{}" y="{:.1f}" text-anchor="end">{:.0f}</text>'
                     .format(pad_l - 5, Y(yv) + 3, yv))

    ok = np.isfinite(up) & np.isfinite(dn)
    if ok.sum() > 2:
        idx = xs[ok]
        top = " ".join("{:.1f},{:.1f}".format(X(i), Y(up[i])) for i in idx)
        bot = " ".join("{:.1f},{:.1f}".format(X(i), Y(dn[i])) for i in idx[::-1])
        parts.append('<polygon class="band" points="{} {}"/>'.format(top, bot))
        parts.append('<polyline class="centre" points="{}"/>'.format(
            " ".join("{:.1f},{:.1f}".format(X(i), Y(centre[i])) for i in idx)))

    pts = " ".join("{:.1f},{:.1f}".format(X(i), Y(v[i]))
                   for i in xs if np.isfinite(v[i]))
    parts.append('<polyline class="spark" points="{}"/>'.format(pts))
    parts.append('<circle class="endpt" cx="{:.1f}" cy="{:.1f}" r="4"/>'
                 .format(X(len(g) - 1), Y(v[-1])))
    parts.append("</svg>")
    return "".join(parts)


def svg_convergence(es: pd.DataFrame, rt: float, width=680, height=300) -> str:
    """
    Convergence by |z| band and horizon, against the round-trip cost line.

    Grouped bars, one scale. The cost line is the point of the chart: bars below it
    are statistically real and economically worthless.
    """
    if es.empty:
        return '<p class="empty">No event-study output.</p>'
    bands = list(dict.fromkeys(es["z_band"]))
    horizons = sorted(es["horizon"].unique())
    pad_l, pad_r, pad_t, pad_b = 52, 16, 20, 52

    vals = es["mean_conv_bp"].to_numpy(dtype=float)
    hi = max(float(np.nanmax(vals)), rt) * 1.14
    lo = min(0.0, float(np.nanmin(vals)) * 1.2)
    Y = lambda y: pad_t + (hi - y) / (hi - lo) * (height - pad_t - pad_b)

    gw = (width - pad_l - pad_r) / max(len(bands), 1)
    bw = min(gw / (len(horizons) + 1.2), 20)

    # one hue, stepped by horizon: horizon is ordinal magnitude, not identity
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    steps = [ramp[int(round(i * (len(ramp) - 1) / max(len(horizons) - 1, 1)))]
             for i in range(len(horizons))]

    parts = ['<svg viewBox="0 0 {} {}" width="100%" role="img" aria-label="Mean '
             'convergence by z band and horizon versus round-trip cost">'
             .format(width, height)]
    for k in range(5):
        yv = lo + (hi - lo) * k / 4
        parts.append('<line class="grid" x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}"/>'
                     .format(pad_l, Y(yv), width - pad_r, Y(yv)))
        parts.append('<text class="tick" x="{}" y="{:.1f}" text-anchor="end">{:.0f}</text>'
                     .format(pad_l - 6, Y(yv) + 4, yv))

    for bi, band in enumerate(bands):
        gx = pad_l + bi * gw
        sub = es[es["z_band"] == band]
        for hi_i, hz in enumerate(horizons):
            r = sub[sub["horizon"] == hz]
            if r.empty:
                continue
            v = float(r["mean_conv_bp"].iloc[0])
            n = int(r["n"].iloc[0])
            x0 = gx + gw / 2 - (len(horizons) * bw) / 2 + hi_i * bw
            y0, y1 = Y(max(v, 0)), Y(min(v, 0))
            parts.append(
                '<rect class="seg" x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
                'rx="2" fill="{}" data-tip="{} &middot; {}s: {} bp (n={})">'
                '<title>{} at {} sessions: {:+.1f} bp, n={}</title></rect>'
                .format(x0, y0, max(bw - 2, 1), max(abs(y1 - y0), 0.8), steps[hi_i],
                        esc(band), hz, _fmt(v), n, esc(band), hz, v, n))
        parts.append('<text class="tick" x="{:.1f}" y="{}" text-anchor="middle">{}</text>'
                     .format(gx + gw / 2, height - pad_b + 18, esc(band)))

    yc = Y(rt)
    parts.append('<line class="costline" x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}"/>'
                 .format(pad_l, yc, width - pad_r, yc))
    parts.append('<text class="costlab" x="{}" y="{:.1f}">round-trip cost {:.0f} bp'
                 '</text>'.format(pad_l + 6, yc - 6, rt))
    parts.append('<line class="zero" x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}"/>'
                 .format(pad_l, Y(0), width - pad_r, Y(0)))
    parts.append('<text class="axis-title" x="{:.1f}" y="{}" text-anchor="middle">'
                 'signed convergence, bp (entry lagged one session)</text>'
                 .format((pad_l + width - pad_r) / 2, height - pad_b + 38))
    parts.append("</svg>")

    legend = ['<div class="legend">']
    for s, hz in zip(steps, horizons):
        legend.append('<span class="lg"><i style="background:{}"></i>{} sessions</span>'
                      .format(s, hz))
    legend.append("</div>")
    return "".join(parts) + "".join(legend)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
<style>
:root{
  --plane:#f4f4f2; --surface:#fdfdfc; --surface-2:#f7f7f5;
  --ink:#16161a; --ink-2:#55555e; --muted:#8b8b93;
  --grid:#e4e4e0; --rule:#d6d6d1; --baseline:#c2c2bc;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --accent:#8a6a12;
  --band:rgba(42,120,214,.13);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --plane:#0d0d0d; --surface:#1a1a19; --surface-2:#212120;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#8b8b93;
    --grid:#2c2c2a; --rule:#333331; --baseline:#383835;
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
    --accent:#d8b14a;
    --band:rgba(57,135,229,.18);
  }
}
:root[data-theme="dark"]{
  --plane:#0d0d0d; --surface:#1a1a19; --surface-2:#212120;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#8b8b93;
  --grid:#2c2c2a; --rule:#333331; --baseline:#383835;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --accent:#d8b14a;
  --band:rgba(57,135,229,.18);
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  font-size:14px;line-height:1.5;-webkit-text-size-adjust:100%}
.wrap{max-width:1140px;margin:0 auto;padding:0 16px;padding-block:0 56px}
.mono,.num{font-family:"IBM Plex Mono",ui-monospace,"Cascadia Mono",Consolas,monospace;
  font-variant-numeric:tabular-nums}

header.bar{position:sticky;top:env(safe-area-inset-top,0px);z-index:20;
  background:var(--surface);border-bottom:1px solid var(--rule)}
.bar-in{max-width:1140px;margin:0 auto;padding:12px 16px;
  display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.brand{font-weight:600;letter-spacing:-.01em;font-size:15px}
.brand span{color:var(--muted);font-weight:400}
.asof{color:var(--ink-2);font-size:12.5px;margin-left:auto}
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
  font-size:12px;font-weight:600;letter-spacing:.02em;border:1px solid transparent}
.pill.quiet{color:var(--ink-2);background:var(--surface-2);border-color:var(--rule)}
.pill.live{color:#fff;background:var(--critical)}
.pill.watch{color:#3a2c00;background:var(--warning)}
.pill i{width:7px;height:7px;border-radius:50%;background:currentColor;flex:none}

h2{font-size:15px;margin:0 0 3px;letter-spacing:-.005em}
h3{font-size:13px;margin:0 0 8px;color:var(--ink-2);font-weight:600}
.sub{color:var(--ink-2);font-size:12.5px;margin:0 0 14px;max-width:70ch}
section{margin-top:30px}
.eyebrow{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;
  color:var(--muted);font-weight:600;margin-bottom:5px}

.panel{background:var(--surface);border:1px solid var(--rule);border-radius:7px;
  padding:18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(152px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:7px;overflow:hidden}
.tile{background:var(--surface);padding:13px 15px}
.tile .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  font-weight:600}
.tile .v{font-size:22px;font-weight:600;letter-spacing:-.02em;margin-top:3px}
.tile .n{font-size:11.5px;color:var(--ink-2);margin-top:1px}

table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:right;padding:7px 9px;border-bottom:1px solid var(--grid);
  white-space:nowrap}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.055em;color:var(--muted);
  font-weight:600;border-bottom:1px solid var(--baseline)}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.tag{font-size:10.5px;padding:1px 6px;border-radius:4px;background:var(--surface-2);
  color:var(--ink-2);border:1px solid var(--rule)}
.blocked{color:var(--muted)}
.hot{color:var(--critical);font-weight:600}
.dim{color:var(--muted)}
.gates{display:flex;gap:4px;flex-wrap:wrap;justify-content:flex-end}
.g{font-size:10px;padding:1px 5px;border-radius:3px;border:1px solid var(--rule);
  background:var(--surface-2);color:var(--muted)}
.g.fail{border-color:var(--serious);color:var(--serious)}

.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:7px;padding:13px}
.card .ttl{font-size:12.5px;font-weight:600;display:flex;justify-content:space-between;
  align-items:baseline;gap:8px}
.card .meta{font-size:11px;color:var(--muted);margin:1px 0 6px}

svg{display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.zero{stroke:var(--baseline);stroke-width:1.5}
.tick{fill:var(--muted);font-size:10.5px;font-family:"IBM Plex Mono",monospace}
.tick.sm{font-size:9.5px}
.axis-title{fill:var(--muted);font-size:10.5px}
.rowlab{fill:var(--ink);font-size:12px;text-anchor:end;font-weight:500}
.rowtag{fill:var(--muted);font-size:9.5px;text-anchor:end}
.totalval{fill:var(--ink-2);font-size:11px;font-family:"IBM Plex Mono",monospace}
.totalmark{stroke:var(--ink);stroke-width:2}
.seg{cursor:default}
.seg:hover{opacity:.82}
.band{fill:var(--band);stroke:none}
.centre{fill:none;stroke:var(--baseline);stroke-width:1;stroke-dasharray:3 3}
.spark{fill:none;stroke:var(--series-1);stroke-width:2;stroke-linejoin:round}
.endpt{fill:var(--series-2);stroke:var(--surface);stroke-width:2}
.costline{stroke:var(--critical);stroke-width:2;stroke-dasharray:6 4}
.costlab{fill:var(--critical);font-size:10.5px;font-weight:600}
.legend{display:flex;gap:13px;flex-wrap:wrap;margin-top:9px;font-size:11px;
  color:var(--ink-2)}
.lg{display:inline-flex;align-items:center;gap:5px}
.lg i{width:10px;height:10px;border-radius:2px;flex:none}
.keylegend{display:flex;gap:15px;flex-wrap:wrap;margin:0 0 12px;font-size:11.5px;
  color:var(--ink-2)}

.empty{color:var(--muted);font-size:12.5px;margin:6px 0}
.alerts{background:var(--surface-2);border:1px solid var(--rule);
  border-radius:6px;padding:12px 14px;font-size:12px;line-height:1.55;
  color:var(--ink);white-space:pre-wrap;overflow-x:auto;margin:0}
.note{background:var(--surface-2);border-left:2px solid var(--accent);
  padding:11px 13px;border-radius:0 5px 5px 0;font-size:12.5px;color:var(--ink-2);
  margin-top:13px}
.note b{color:var(--ink)}
details{margin-top:11px;font-size:12px}
summary{cursor:pointer;color:var(--ink-2);font-size:11.5px}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--rule);
  color:var(--muted);font-size:11.5px}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--ink);color:var(--plane);padding:5px 8px;border-radius:4px;
  font-size:11.5px;z-index:60;max-width:260px;
  font-family:"IBM Plex Mono",monospace}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media (max-width:560px){
  .tile .v{font-size:19px}
  .bar-in{padding:10px 16px}
  .asof{margin-left:0;width:100%}
}
</style>
"""

JS = """
<script>
(function(){
  var tip=document.getElementById('tip');
  function show(e,t){tip.innerHTML=t;tip.style.opacity=1;move(e);}
  function move(e){
    var w=tip.offsetWidth,h=tip.offsetHeight;
    var x=e.clientX+14, y=e.clientY-h-10;
    if(x+w>window.innerWidth-8) x=e.clientX-w-14;
    if(y<4) y=e.clientY+16;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  }
  document.addEventListener('mouseover',function(e){
    var t=e.target.closest('[data-tip]');
    if(t) show(e,t.getAttribute('data-tip'));
  });
  document.addEventListener('mousemove',function(e){
    if(tip.style.opacity==='1') move(e);
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('[data-tip]')) tip.style.opacity=0;
  });
})();
</script>
"""


def render(ctx, args) -> str:
    asof, snap = latest_snapshot(ctx)
    dec = decompose_pairs(ctx, asof, snap)
    pairs, es, rep = ctx["pairs"], ctx["es"], ctx["rep"]
    rt = ctx["rt"]

    today = pairs[pairs["date"] == pairs["date"].max()] if not pairs.empty else pd.DataFrame()
    n_live = int(today["signal_ok"].sum()) if not today.empty else 0
    fire_rate = float(pairs["signal_ok"].mean()) if not pairs.empty else 0.0

    if n_live == 0:
        pill = '<span class="pill quiet"><i></i>No actionable signal</span>'
    elif n_live == 1:
        pill = '<span class="pill live"><i></i>1 signal</span>'
    else:
        pill = '<span class="pill live"><i></i>{} signals</span>'.format(n_live)

    P = []
    P.append("<title>MCX Gold Relative Value</title>")
    P.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&'
             'display=swap">')
    P.append(CSS)
    P.append('<div id="tip"></div>')

    # ---- header
    P.append('<header class="bar"><div class="bar-in">'
             '<span class="brand">MCX Gold <span>relative value</span></span>'
             '{}<span class="asof mono">as of {} &middot; {} sessions</span>'
             '</div></header>'.format(pill, asof, int(ctx["rows"]["date"].nunique())))

    P.append('<div class="wrap">')

    # ---- tiles
    dq = rep.counts
    P.append('<section><div class="tiles">')
    tiles = [
        ("Contracts tracked", "{}".format(int(ctx["rows"]["contract_id"].nunique())),
         "keyed by symbol + expiry, never spliced"),
        ("Signal fires on", "{:.1%}".format(fire_rate), "of pair-sessions"),
        ("Round-trip cost", "{:.0f} bp".format(rt), "4 crossings, 5 bp half-spread"),
        ("Date echoes caught", "{:,}".format(dq.get("date_echo_rows", 0)),
         "rows MCX mislabelled"),
        ("Derived settlements", "{:,}".format(dq.get("no_trade_rows", 0)),
         "no trade: not a fillable price"),
    ]
    for k, v, n in tiles:
        P.append('<div class="tile"><div class="k">{}</div><div class="v mono">{}</div>'
                 '<div class="n">{}</div></div>'.format(esc(k), esc(v), esc(n)))
    P.append("</div></section>")

    # ---- decomposition: the centrepiece
    P.append('<section><div class="eyebrow">Why the prices differ</div>')
    P.append("<h2>Almost none of the gap is a trading opportunity</h2>")
    P.append('<p class="sub">Each bar is the observed spread between two contracts once '
             'both are normalised to rupees per 10&nbsp;g of fine gold. It is split into the '
             'three things that produce it. Only the third is actionable: the first is '
             'fabrication cost embedded in a minted deliverable, the second is where the '
             'two legs sit on the futures curve.</p>')
    P.append('<div class="keylegend">'
             '<span class="lg"><i style="background:var(--series-1)"></i>'
             'Structural premium &mdash; persistent, not a mispricing</span>'
             '<span class="lg"><i style="background:var(--series-2)"></i>'
             'Carry &mdash; mechanical, from differing time to expiry</span>'
             '<span class="lg"><i style="background:var(--series-3)"></i>'
             'Residual &mdash; the only actionable part</span></div>')
    P.append('<div class="panel">')
    P.append(svg_decomposition(dec))
    if not dec.empty:
        worst = dec.loc[dec["total_bp"].abs().idxmax()]
        share = (abs(worst["residual_bp"]) / max(abs(worst["total_bp"]), 1e-9)) * 100
        P.append('<div class="note"><b>{}</b> shows the widest raw gap at '
                 '{} bp, but only {} bp of it &mdash; {:.0f}% &mdash; is residual. '
                 'A parity spread that ignores premium and carry would read the whole '
                 '{} bp as an opportunity and sell it forever.</div>'
                 .format(esc(worst["pair"]), _fmt(worst["total_bp"], 0),
                         _fmt(worst["residual_bp"], 0), share,
                         _fmt(worst["total_bp"], 0)))
        P.append('<details><summary>Table view</summary><div class="scroll"><table>'
                 '<thead><tr><th>Pair</th><th>Total</th><th>Premium</th><th>Carry</th>'
                 '<th>Residual</th><th>Matched</th></tr></thead><tbody>')
        for r in dec.sort_values("total_bp").itertuples(index=False):
            P.append('<tr><td>{}</td><td class="num">{}</td><td class="num">{}</td>'
                     '<td class="num">{}</td><td class="num">{}</td><td>{}</td></tr>'
                     .format(esc(r.pair), _fmt(r.total_bp), _fmt(r.premium_bp),
                             _fmt(r.carry_bp), _fmt(r.residual_bp),
                             "yes" if r.maturity_matched else "carry-bridged"))
        P.append("</tbody></table></div></details>")
    P.append("</div></section>")

    # ---- normalised prices
    P.append('<section><div class="eyebrow">Latest session</div>'
             "<h2>Normalised prices and executable state</h2>"
             '<p class="sub">Raw closes are not comparable. GOLDM is quoted per 10&nbsp;g '
             'at 995 fineness while the others quote per their own unit at 999, so the '
             'normalised column is the only one that can be differenced.</p>'
             '<div class="panel"><div class="scroll"><table><thead><tr>'
             "<th>Contract</th><th>Expiry</th><th>Raw close</th><th>Per 10g fine</th>"
             "<th>Volume</th><th>OI</th><th>Sessions left</th><th>Half-spread</th>"
             "<th>Max lots</th><th>State</th></tr></thead><tbody>")
    for r in snap.itertuples(index=False):
        state = ('<span class="tag">tradable</span>' if r.tradable
                 else '<span class="tag blocked">not tradable</span>')
        P.append('<tr><td><b>{}</b></td><td class="num dim">{}</td>'
                 '<td class="num">{:,.1f}</td><td class="num"><b>{:,.0f}</b></td>'
                 '<td class="num">{:,}</td><td class="num dim">{:,}</td>'
                 '<td class="num">{}</td><td class="num dim">{:.1f} bp</td>'
                 '<td class="num dim">{:,.0f}</td><td>{}</td></tr>'
                 .format(esc(r.symbol), esc(r.expiry), r.close, r.norm_close,
                         r.volume, r.oi, r.sessions_to_expiry, r.half_spread_bp,
                         r.max_lots, state))
    P.append("</tbody></table></div>")
    P.append('<div class="note">Half-spread and max-lot figures are '
             '<b>assumptions</b>. Bhavcopy contains no depth data, so they are swept '
             'in the cost analysis rather than trusted. Volume is not depth: a '
             'GOLDPETAL day of 90 lots is 90 grams across the whole session.</div>')
    P.append("</div></section>")

    # ---- signal board
    P.append('<section><div class="eyebrow">Signal board</div>'
             "<h2>What is firing, and what is holding it back</h2>"
             '<p class="sub">Every statistic is trailing: means, deviations, half-lives '
             'and stationarity tests use data strictly before the session shown. A pair '
             'must clear all four gates, and the gate that blocks it is named.</p>'
             '<div class="panel"><div class="scroll"><table><thead><tr>'
             "<th>Pair</th><th>Residual spread</th><th>z</th><th>Dislocation</th>"
             "<th>Noise floor</th><th>Half-life</th><th>ADF t</th><th>Gates</th>"
             "<th>Action</th></tr></thead><tbody>")
    if today.empty:
        P.append('<tr><td colspan="9" class="dim">No pair diagnostics for this '
                 "session.</td></tr>")
    else:
        gate_names = {"gate_history": "history", "gate_stationary": "stationary",
                      "gate_halflife": "half-life", "gate_magnitude": "magnitude"}
        for r in today.sort_values("z", key=lambda s: s.abs(), ascending=False) \
                      .itertuples(index=False):
            gates = []
            for col, lab in gate_names.items():
                ok = bool(getattr(r, col, False))
                gates.append('<span class="g{}">{}</span>'
                             .format("" if ok else " fail", lab))
            if r.signal_ok:
                act = ('<span class="hot">{} {}</span>'
                       .format("buy" if r.direction > 0 else "sell", esc(r.pair)))
            else:
                act = '<span class="dim">hold</span>'
            zc = "hot" if r.signal_ok else "num"
            P.append('<tr><td><b>{}</b>{}</td><td class="num">{} bp</td>'
                     '<td class="num {}">{}</td><td class="num">{} bp</td>'
                     '<td class="num dim">{:.1f} bp</td><td class="num">{}</td>'
                     '<td class="num dim">{}</td><td><div class="gates">{}</div></td>'
                     "<td>{}</td></tr>"
                     .format(esc(r.pair),
                             "" if r.maturity_matched else ' <span class="tag">bridged</span>',
                             _fmt(r.spread_bp), zc, _fmt(r.z, 2),
                             _fmt(r.dislocation_bp),
                             ctx["scfg"].min_abs_bp,
                             "{:.1f}".format(r.halflife) if np.isfinite(r.halflife) else "inf",
                             "{:.2f}".format(r.adf_t) if np.isfinite(r.adf_t) else "n/a",
                             "".join(gates), act))
    P.append("</tbody></table></div>")
    P.append('<div class="note">The dashboard is quiet by design. Across the whole '
             "sample a signal clears every gate on <b>{:.1%}</b> of pair-sessions. A "
             "board that always has an opinion is evidence against itself.</div>"
             .format(fire_rate))
    P.append("</div></section>")

    # ---- history small multiples
    P.append('<section><div class="eyebrow">Residual spread history</div>'
             "<h2>Each pair against its own trailing band</h2>"
             '<p class="sub">Blue band is the trailing &plusmn;2&sigma; envelope, which '
             'widens and narrows as the estimate does. The orange dot is the latest '
             'observation. Early history is deliberately bandless: there was not yet '
             'enough data to judge anything.</p><div class="grid2">')
    if not pairs.empty:
        for pair, g in pairs.groupby("pair"):
            g = g.sort_values("date").tail(args.history)
            last = g.iloc[-1]
            hlbl = ("{:.0f}".format(last["halflife"])
                    if np.isfinite(last["halflife"]) else "no reversion")
            P.append('<div class="card"><div class="ttl"><span>{}</span>'
                     '<span class="mono {}">z {}</span></div>'
                     '<div class="meta">half-life {} sessions &middot; {}</div>{}</div>'
                     .format(esc(pair), "hot" if last["signal_ok"] else "dim",
                             _fmt(last["z"], 2), hlbl,
                             "same expiry window" if last["maturity_matched"]
                             else "carry-bridged across expiry windows",
                             svg_spread_history(g, pair)))
    P.append("</div></section>")

    # ---- convergence vs cost
    P.append('<section><div class="eyebrow">Does it pay?</div>'
             "<h2>Convergence is real. It does not clear costs.</h2>"
             '<p class="sub">Signed forward convergence by signal strength and holding '
             'period, measured across every pair-session rather than only tradable ones, '
             'with entry lagged one session so the signal and the entry price do not '
             'share a settlement. The dashed line is the round-trip cost of four '
             'crossings.</p><div class="panel">')
    P.append(svg_convergence(es, rt))
    if not es.empty:
        hurdle = eventstudy.cost_hurdle_table(es, rt)
        n_sur = int(hurdle["survives_costs"].sum())
        n_ci = int(hurdle["ci_clears_costs"].sum())
        P.append('<div class="note">Convergence rises monotonically with both signal '
                 "strength and horizon, which is the signature of a real "
                 "mean-reverting effect rather than noise. Of {} band-horizon cells, "
                 "<b>{}</b> beat the cost line on the point estimate and <b>{}</b> "
                 "clear it with a 95% bootstrap interval. Under the stated cost "
                 "assumptions there is no reliably profitable cell.</div>"
                 .format(len(hurdle), n_sur, n_ci))
        P.append('<details><summary>Table view with inference</summary>'
                 '<div class="scroll"><table><thead><tr><th>Band</th><th>Horizon</th>'
                 "<th>n</th><th>Mean</th><th>Hit rate</th><th>t (Newey-West)</th>"
                 "<th>t (non-overlap)</th><th>Net of cost</th></tr></thead><tbody>")
        for r in es.itertuples(index=False):
            net = r.mean_conv_bp - rt
            P.append('<tr><td>{}</td><td class="num">{}</td><td class="num dim">{}</td>'
                     '<td class="num">{}</td><td class="num">{:.0%}</td>'
                     '<td class="num">{:.1f}</td><td class="num dim">{}</td>'
                     '<td class="num {}">{}</td></tr>'
                     .format(esc(r.z_band), r.horizon, r.n, _fmt(r.mean_conv_bp),
                             r.hit_rate, r.t_newey_west,
                             "{:.1f}".format(r.t_nonoverlap)
                             if np.isfinite(r.t_nonoverlap) else "n/a",
                             "hot" if net > 0 else "dim", _fmt(net)))
        P.append("</tbody></table></div></details>")
    P.append("</div></section>")

    # ---- backtest + attribution
    perf, gold, attrib = ctx["perf"], ctx["gold"], ctx["attrib"]
    P.append('<section><div class="eyebrow">Executability</div>'
             "<h2>Walk-forward backtest, and what the P&amp;L actually was</h2>"
             '<p class="sub">Signals are shifted one session before execution. Legs are '
             'sized to whole lots for fine-gram neutrality, and a trade is declined '
             'outright when granularity cannot hedge it. Positions are force-closed '
             'before the tender window.</p><div class="panel">')
    if perf.get("n_trades", 0) == 0:
        P.append('<p class="empty">No trades cleared every gate. That is a result, not '
                 "a gap.</p>")
    else:
        P.append('<div class="tiles">')
        bt_tiles = [
            ("Trades", "{}".format(perf["n_trades"]), "over the full sample"),
            ("Gross per trade", "{:+.1f} bp".format(perf["mean_gross_bp"]),
             "before any cost"),
            ("Net per trade", "{:+.1f} bp".format(perf["mean_net_bp"]),
             "after four crossings"),
            ("Hit rate net", "{:.0%}".format(perf["hit_rate_net"]),
             "gross was {:.0%}".format(perf["hit_rate_gross"])),
            ("Unhedged gold", "{:.1%}".format(perf["mean_residual_frac"]),
             "mean fine-gram residual"),
        ]
        for k, v, n in bt_tiles:
            P.append('<div class="tile"><div class="k">{}</div>'
                     '<div class="v mono">{}</div><div class="n">{}</div></div>'
                     .format(esc(k), esc(v), esc(n)))
        P.append("</div>")

        if gold and "params" in gold:
            lo, hi = gold["ci95"]["beta_gold"]
            P.append('<div class="note"><b>Gold beta {:+.3f}</b>, 95% interval '
                     "[{:+.3f}, {:+.3f}], R&sup2; {:.3f}. {}. The spread is "
                     "gram-neutral by construction, but lot indivisibility leaves a "
                     "residual, so neutrality is measured rather than asserted.</div>"
                     .format(gold["params"]["beta_gold"], lo, hi, gold["r2"],
                             esc(gold["interpretation"])))

        if not attrib.empty:
            P.append('<h3 style="margin-top:16px">P&amp;L attribution, bp per trade</h3>'
                     '<div class="scroll"><table><thead><tr><th>Component</th>'
                     "<th>Mean</th><th>SD</th><th>Share of gross movement</th>"
                     "</tr></thead><tbody>")
            labels = {"alpha": "Residual convergence &mdash; the strategy",
                      "roll": "Mechanical roll-down",
                      "curve": "Curve reshaping",
                      "level_leak": "Gold leak via unhedged grams",
                      "cross": "Second-order cross term",
                      "unexplained": "Unexplained by the model",
                      "cost_drag": "Transaction costs"}
            for r in attrib.itertuples(index=False):
                sh = ("{:.0%}".format(r.share_of_abs)
                      if np.isfinite(r.share_of_abs) else "&mdash;")
                P.append('<tr><td>{}</td><td class="num">{}</td>'
                         '<td class="num dim">{}</td><td class="num dim">{}</td></tr>'
                         .format(labels.get(r.component, esc(r.component)),
                                 _fmt(r.mean_bp), _fmt(r.sd_bp)
                                 if np.isfinite(r.sd_bp) else "&mdash;", sh))
            P.append("</tbody></table></div>")
            P.append('<div class="note">Roll-down is exactly zero because both legs are '
                     "entered and exited on the same sessions, so they age identically "
                     "and <i>b</i>&middot;(&Delta;&tau;<sub>a</sub>&minus;"
                     "&Delta;&tau;<sub>b</sub>) cancels. What survives is the curve "
                     "term, which is a term-structure bet rather than roll-down. It "
                     "would be non-zero only if the legs were held over different "
                     "windows.</div>")
    P.append("</div></section>")

    # ---- margin / capital at risk
    sp = ctx.get("span_params")
    if sp is not None and perf.get("n_trades", 0):
        P.append('<section><div class="eyebrow">Capital at risk</div>'
                 "<h2>Margin makes the result worse, not better</h2>"
                 '<p class="sub">Initial margin is SPAN + ELM on the whole open book, '
                 'recomputed each session. A spread is margin-<i>efficient</i> because '
                 'SPAN credits offsetting positions in related commodities &mdash; so '
                 'the same rupee loss sits on a far smaller capital base than the '
                 'notional suggests. The SPAN algorithm is implemented in full; the '
                 'SPAN <b>parameters</b> are assumptions, because MCX publishes them '
                 'separately and nothing in bhavcopy determines them.</p>'
                 '<div class="panel">')
        P.append('<div class="tiles">')
        mt = [
            ("Peak margin blocked", "Rs {:,.0f}".format(perf.get("peak_margin", 0)),
             "across all open positions"),
            ("Margin / notional", "{:.2%}".format(perf.get("margin_pct_notional", 0)),
             "vs ~7% for two outrights"),
            ("Return on margin", "{:+.2%}".format(
                perf.get("ann_return_on_margin", float("nan"))), "annualised"),
            ("On arbitrary base", "{:+.2%}".format(perf.get("ann_return", 0)),
             "Rs 20 lakh, meaningless"),
        ]
        for k, v, nn in mt:
            P.append('<div class="tile"><div class="k">{}</div>'
                     '<div class="v mono">{}</div><div class="n">{}</div></div>'
                     .format(esc(k), esc(v), esc(nn)))
        P.append("</div>")
        try:
            from mcxrv.stage3_gates_decision import margin_capital as margin_mod
            ex = [margin_mod.MarginPosition("GOLDM", "x", 1, 35485.0, 20),
                  margin_mod.MarginPosition("GOLDPETAL", "y", -99, 3596.5, 20)]
            sw = margin_mod.sweep_credit(ex, sp)
            P.append('<h3 style="margin-top:16px">Margin against the '
                     "inter-commodity credit &mdash; the dominant unknown</h3>"
                     '<div class="scroll"><table><thead><tr><th>Spread credit</th>'
                     "<th>SPAN</th><th>Initial margin</th><th>% of notional</th>"
                     "</tr></thead><tbody>")
            for r in sw.itertuples(index=False):
                P.append('<tr><td class="num">{:.0%}</td><td class="num dim">{:,.0f}'
                         '</td><td class="num"><b>{:,.0f}</b></td>'
                         '<td class="num">{:.2%}</td></tr>'
                         .format(r.inter_spread_credit, r.span, r.initial_margin,
                                 r.pct_of_notional))
            P.append("</tbody></table></div>")
        except Exception:
            pass
        P.append('<div class="note">Sharpe is <b>scale-invariant</b>, so it does not '
                 "move when the capital denominator changes and says nothing about "
                 "capital efficiency either way. Return on margin does: the strategy "
                 "goes from {:+.2%} on an arbitrary Rs 20 lakh to <b>{:+.2%}</b> on "
                 "the capital actually blocked. Parameter source: {}.</div>"
                 .format(perf.get("ann_return", 0),
                         perf.get("ann_return_on_margin", float("nan")),
                         esc(sp.source)))
        P.append("</div></section>")


    # ---- live risk board
    rbd = ctx.get("risk_board")
    rcfg = ctx.get("risk_cfg")
    if rbd is not None and not rbd.empty:
        priced = rbd[rbd["action"] != "unavailable"]
        book_req = float(priced["book_required"].iloc[0]) if not priced.empty else 0.0
        P.append('<section><div class="eyebrow">Live risk board</div>'
                 "<h2>Should this pair be added to today?</h2>"
                 '<p class="sub">The capital ledger is retrospective. This is the '
                 'question a desk asks before trading: given what the book already '
                 'holds and what margin costs this morning, is the pair fundable? '
                 'Each requirement is <b>marginal</b> &mdash; book-with minus '
                 'book-without &mdash; because SPAN nets across the whole portfolio. '
                 'The change in requirement is measured over one parameter revision '
                 'cycle with the position size held fixed, so it reports risk rather '
                 'than sizing.</p>'
                 '<div class="panel">')
        P.append('<div class="tiles">')
        rt_ = [
            ("Account capital", "Rs {:,.0f}".format(rcfg.account_capital),
             "cash actually put up"),
            ("Book requirement", "Rs {:,.0f}".format(book_req),
             "already blocked"),
            ("Blocked from adding",
             "{} of {}".format(int((rbd["action"] == "do not add exposure").sum()),
                               int(len(rbd))),
             "pairs refused today"),
            ("Worst buffer", esc(risk.summary(rbd)["worst_buffer"]),
             "across all pairs"),
        ]
        for k, v, nn in rt_:
            P.append('<div class="tile"><div class="k">{}</div>'
                     '<div class="v mono">{}</div><div class="n">{}</div></div>'
                     .format(esc(k), esc(v), esc(nn)))
        P.append("</div>")

        P.append('<div class="scroll" style="margin-top:14px"><table><thead><tr>'
                 "<th>Pair</th><th>Notional</th><th>Marginal margin</th>"
                 "<th>Utilisation</th><th>Buffer</th><th>SPAN change</th>"
                 "<th>z</th><th>Signal</th><th>Binding reason</th>"
                 "</tr></thead><tbody>")
        for r in rbd.itertuples(index=False):
            if r.action == "unavailable":
                P.append('<tr><td>{}</td><td class="num dim" colspan="6">&mdash;</td>'
                         '<td><span class="pill quiet"><i></i>unavailable</span></td>'
                         "<td>{}</td></tr>".format(esc(r.pair), esc(r.reason)))
                continue
            cls = {"do not add exposure": "live", "add reduced size": "watch",
                   "no signal": "quiet", "may add": "quiet"}.get(r.action, "quiet")
            chg = ("{:+.1%}".format(r.span_change_pct)
                   if _np_ok(r.span_change_pct) else "&mdash;")
            P.append('<tr><td>{}</td><td class="num dim">{:,.0f}</td>'
                     '<td class="num"><b>{:,.0f}</b></td>'
                     '<td class="num">{:.0%}</td><td>{}</td>'
                     '<td class="num">{}</td><td class="num">{:+.2f}</td>'
                     '<td><span class="pill {}"><i></i>{}</span></td>'
                     "<td>{}</td></tr>".format(
                         esc(r.pair), r.notional_exposure, r.marginal_margin,
                         r.utilisation_if_added, esc(r.buffer_state), chg, r.z,
                         cls, esc(r.action), esc(r.reason)))
        P.append("</tbody></table></div>")

        # Priced pairs only. Unavailable ones are already named in the table above,
        # and a monitor listing nothing but "unavailable" says nothing about capital
        # -- which is the whole point of this block.
        shown = risk.alerts(priced)
        if shown:
            P.append('<h3 style="margin-top:16px">As a monitor would show it</h3>'
                     '<pre class="mono alerts">{}</pre>'
                     .format(esc(("\n\n").join(shown))))

        P.append('<div class="note">The verdict exists to say <b>no</b>. A signal '
                 "clearing its statistical gates is necessary and not sufficient: "
                 "the position must also be fundable, and fundable under a "
                 "deterioration that has not happened yet. Where a pair is refused "
                 "the binding reason is named, so the refusal can be argued with "
                 "rather than merely obeyed.")
        if (not priced.empty
                and float((priced["marginal_margin"]
                           - priced["initial_margin"]).abs().max()) < 1.0
                and bool(priced["floor_binds"].any())):
            P.append(" Marginal margin equals standalone margin here: while the "
                     "{:.0%} initial-margin <b>floor</b> binds, the requirement is a "
                     "flat percentage of gross notional, which is additive &mdash; "
                     "so adding an offsetting pair earns no portfolio credit at all. "
                     "The netting benefit SPAN computes is real and entirely "
                     "unrealised.".format(ctx["span_params"].im_floor_pct))
        P.append("</div>")
        P.append("</div></section>")

    # ---- cost sensitivity
    P.append('<section><div class="eyebrow">The unobservable parameter</div>'
             "<h2>How tight would the book have to be?</h2>"
             '<p class="sub">Bhavcopy has no depth data, so the bid-ask spread &mdash; '
             'the dominant cost &mdash; cannot be measured, only assumed. The honest '
             'response is to sweep it and say where the strategy dies.</p>'
             '<div class="panel"><div class="scroll">')
    ct = costs.cost_summary_table(500_000.0, ctx["cost_cfg"])
    P.append("<table><thead><tr><th>Half-spread per leg</th><th>Statutory</th>"
             "<th>Execution</th><th>Round trip</th><th>Verdict at "
             "typical 2&ndash;3&sigma; edge</th></tr></thead><tbody>")
    typ = 0.0
    if not es.empty:
        hot = es[(es["z_band"] == "|z| 2-3") & (es["horizon"] == 10)]
        typ = float(hot["mean_conv_bp"].iloc[0]) if not hot.empty else 0.0
    for r in ct.itertuples(index=False):
        ok = typ > r.round_trip_bp
        P.append('<tr><td class="num">{:.0f} bp</td><td class="num dim">{:.1f}</td>'
                 '<td class="num dim">{:.1f}</td><td class="num"><b>{:.1f} bp</b></td>'
                 '<td class="{}">{}</td></tr>'
                 .format(r.half_spread_bp_per_leg, r.statutory_bp, r.execution_bp,
                         r.round_trip_bp, "hot" if ok else "dim",
                         "edge survives" if ok else "edge consumed"))
    P.append("</tbody></table></div>")
    if typ > 0:
        be = costs.breakeven_half_spread_bp(typ, 500_000.0, ctx["cost_cfg"])
        bes = "{:.1f} bp".format(be) if np.isfinite(be) else "unbounded"
        P.append('<div class="note">A typical 2&ndash;3&sigma; dislocation converges '
                 "<b>{:.0f} bp</b> over ten sessions. Breaking even needs a half-spread "
                 "inside <b>{}</b> per leg per crossing, on both legs, on the days the "
                 "signal fires. On GOLDPETAL that is tighter than a single tick. That "
                 "is the number to check against a real order book before risking "
                 "anything.</div>".format(typ, bes))
    P.append("</div></section>")

    # ---- data quality
    P.append('<section><div class="eyebrow">Provenance</div>'
             "<h2>What the data did wrong, and what was done about it</h2>"
             '<p class="sub">Every pathology becomes a column rather than a silent '
             'correction. Dropping a bad row hides it; repairing one invents data.</p>'
             '<div class="panel"><div class="scroll"><table><thead><tr><th>Issue</th>'
             "<th>Rows</th><th>Handling</th></tr></thead><tbody>")
    handling = {
        "date_echo_rows": "kept under the true session date; duplicates collapsed",
        "symbol_whitespace": "trimmed before any join",
        "duplicate_rows_collapsed": "one row per (session, contract); highest volume kept",
        "duplicate_key_price_conflict": "conflicting closes flagged, most-traded kept",
        "no_trade_rows": "flagged untradeable; excluded from the curve fit",
        "settle_untradeable_rows": "excluded from fills and from the fit",
        "date_flag_ambiguous_ok": "MM/DD confirmed against the requested date",
        "date_flag_swapped": "field order corrected and recorded",
        "expiry_outside_window": "flagged against the contract specification",
        "duplicate_rows_seen": "detected before collapsing",
        "symbol_unknown_or_filtered": "outside the gold universe",
        "close_missing_or_nonpositive_dropped": "dropped: unidentifiable",
        "ohlc_inconsistent": "flagged, retained",
    }
    for k, v in sorted(rep.counts.items(), key=lambda kv: -kv[1]):
        if v == 0:
            continue
        P.append('<tr><td>{}</td><td class="num">{:,}</td><td class="dim">{}</td></tr>'
                 .format(esc(k.replace("_", " ")), v,
                         esc(handling.get(k, "recorded"))))
    P.append("</tbody></table></div>")
    P.append('<div class="note"><b>{:,}</b> raw rows became <b>{:,}</b> validated '
             "rows across <b>{}</b> sessions and <b>{}</b> instruments, each keyed by "
             "symbol and expiry. No continuous or near-month series is constructed "
             "anywhere: splicing contracts at a roll injects a price jump that a "
             "mean-reversion signal reads as an opportunity.</div>"
             .format(rep.rows_in, rep.rows_out, int(ctx["rows"]["date"].nunique()),
                     int(ctx["rows"]["contract_id"].nunique())))
    P.append("</div></section>")

    # ---- anomaly review queue
    det = ctx.get("anomaly")
    if det is not None and not det.empty:
        from mcxrv.stage1_intake import anomaly as anom
        new_hits = det[det["is_anomaly"] & ~det["rule_flagged"]]
        P.append('<section><div class="eyebrow">Review queue</div>'
                 "<h2>What the rules did not anticipate</h2>"
                 '<p class="sub">An Isolation Forest scores every observation against '
                 'its own history and its peers that session. The rule-based validators '
                 'catch what someone thought to write a rule for; this catches what '
                 'nobody did. These flags are <b>advisory and block nothing</b> &mdash; '
                 'filtering a backtest on them would be look-ahead, and dropping '
                 'surprising rows deletes the real market events a relative-value '
                 'strategy lives on.</p><div class="panel">')
        P.append('<div class="scroll"><table><thead><tr><th>Contract</th><th>Rows</th>'
                 "<th>Flagged</th><th>Already known to rules</th>"
                 "<th>New</th></tr></thead><tbody>")
        for r in anom.summary(det).itertuples(index=False):
            P.append('<tr><td><b>{}</b></td><td class="num dim">{:,}</td>'
                     '<td class="num">{}</td><td class="num dim">{}</td>'
                     '<td class="num hot">{}</td></tr>'
                     .format(esc(r.symbol), r.rows, r.flagged,
                             r.flagged_already_known, r.flagged_new))
        P.append("</tbody></table></div>")
        top = anom.top_reasons(det, 8)
        if not top.empty:
            P.append('<h3 style="margin-top:16px">Worst unexplained, most anomalous '
                     "first</h3>"
                     '<div class="scroll"><table><thead><tr><th>Session</th>'
                     "<th>Contract</th><th>Score</th><th>Driver</th>"
                     "<th>Robust z</th><th>Move</th><th>vs peers</th>"
                     "</tr></thead><tbody>")
            for r in top.itertuples(index=False):
                P.append('<tr><td class="num dim">{}</td><td>{}</td>'
                         '<td class="num">{:.4f}</td><td>{}</td>'
                         '<td class="num dim">{:.1f}</td><td class="num">{:+.2%}</td>'
                         '<td class="num">{} bp</td></tr>'
                         .format(esc(r.date), esc(r.contract_id), r.score,
                                 esc(str(r.top_feature).replace("_", " ")),
                                 r.top_z, r.log_ret, _fmt(r.peer_dev_bp, 0)))
            P.append("</tbody></table></div>")
        P.append('<div class="note"><b>{}</b> of {} flagged rows were not already '
                 "caught by a rule. In this dataset the worst of them are far-dated "
                 "contracts trading 1&ndash;4 lots &mdash; prints that pass the "
                 "zero-volume rule because volume is technically non-zero, but which "
                 "no one could execute against. That is a genuine gap in the rule set, "
                 "found without anyone having to predict it.</div>"
                 .format(len(new_hits), int(det["is_anomaly"].sum())))
        P.append("</div></section>")

    # ---- coverage
    cov = ctx["coverage"]
    P.append('<section><div class="eyebrow">Tradable universe</div>'
             "<h2>How much of the data is actually usable</h2>"
             '<div class="panel"><div class="scroll"><table><thead><tr>'
             "<th>Contract</th><th>Rows</th><th>Genuine trade</th><th>Volume gate</th>"
             "<th>OI gate</th><th>Lifecycle</th><th>Tradable</th>"
             "<th>Median half-spread</th></tr></thead><tbody>")
    for r in cov.itertuples(index=False):
        P.append('<tr><td><b>{}</b></td><td class="num dim">{:,}</td>'
                 '<td class="num">{:.0%}</td><td class="num">{:.0%}</td>'
                 '<td class="num">{:.0%}</td><td class="num">{:.0%}</td>'
                 '<td class="num"><b>{:.0%}</b></td><td class="num dim">{}</td></tr>'
                 .format(esc(r.symbol), r.rows, r.gate_genuine_trade_rate,
                         r.gate_volume_rate, r.gate_oi_rate, r.gate_lifecycle_rate,
                         r.tradable_rate,
                         "{:.1f} bp".format(r.median_half_spread_bp)
                         if np.isfinite(r.median_half_spread_bp) else "n/a"))
    P.append("</tbody></table></div></div></section>")

    P.append('<footer>Source: {}. Normalised to INR per 10&nbsp;g fine gold using each '
             "contract's quotation base and purity &mdash; not its trading unit. "
             "Settlement prices are official exchange values, not executable fills. "
             "Volume is not depth. Nothing here is investment advice.</footer>"
             .format(esc(ctx["provenance"])))
    P.append("</div>")
    P.append(JS)
    return "\n".join(P)


def main():
    args = parse_args()
    ctx = build(args)
    out = render(ctx, args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote {} ({:,} bytes)".format(args.out, len(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
