import { Link, useParams } from "react-router-dom";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, Scatter, XAxis, YAxis } from "recharts";
import { useOverview, usePair, useRisk, type Account, type Leg, type PairDetail as PD, type Risk } from "../api";
import { bp, compact, fdate, fexp, fin, fmon, inr, ms, num, pct, vs } from "../format";
import { usePalette } from "../theme";
import { Box, SoftTooltip, axis } from "../components/charts";
import { Card, Chip, Legend, Loaded } from "../components/ui";

const ENTRY_SESSIONS = 15; // BacktestConfig.min_sessions_to_expiry_entry

type Check = { k: string; ok: boolean | null; d: string };
function checks(p: PD, r: Risk, acct: Account | undefined): Check[] {
  const g = p.gates || {}, legs: Leg[] = p.legs ?? [];
  const left = Math.min(...legs.map((l) => (fin(l.sessions_to_expiry) ? l.sessions_to_expiry : Infinity)));
  const fundable = fin(r.utilisation_if_added) ? r.utilisation_if_added < 0.8 && !/breached|low/.test(r.buffer_state ?? "") : null;
  // Margin answers "could the account fund it?". The kill switch and "no signal" are
  // other questions, so they fall back to the fundability test here.
  const otherReason = r.action === "no signal" || /^account drawdown/.test(r.reason ?? "");
  const marginOk = /^(may add|add reduced size)$/.test(r.action ?? "") ? true : otherReason ? fundable : r.action ? false : null;
  const accountCheck: Check[] = acct && fin(acct.drawdown_pct) && fin(acct.limit_pct) ? [{
    k: "Account", ok: !acct.halted,
    d: `${pct(acct.drawdown_pct, 1)} below the account's peak, kill switch at ${pct(acct.limit_pct)}` + (acct.halted ? ": new exposure halted until reviewed" : ""),
  }] : [];
  return [...accountCheck,
    { k: "History", ok: g.gate_history ?? null, d: g.gate_history ? "enough trailing sessions to score this pair" : "not enough history to score yet" },
    { k: "Mean-reverting", ok: g.gate_stationary ?? null, d: `ADF t ${num(p.adf_t)} (needs below −2.57)` },
    { k: "Half-life", ok: g.gate_halflife ?? null, d: `${fin(p.halflife) ? num(p.halflife, 1) + " sessions" : "∞"} to close half the gap (limit 30)` },
    { k: "Size", ok: g.gate_magnitude ?? null, d: `|z| ${num(fin(p.z) ? Math.abs(p.z) : null)} vs ${p.z_entry}; extra ${bp(p.dislocation_bp)} vs 15 bp minimum and the noise floor` },
    { k: "Liquidity", ok: legs.length === 2 ? legs.every((l) => l.tradable !== false && l.liquidity !== "RED") : null,
      d: legs.map((l) => `${l.symbol} ${compact(l.avg_volume_20)} lots/day (${l.liquidity ?? "–"})`).join(" · ") },
    { k: "Lifecycle", ok: Number.isFinite(left) ? left >= ENTRY_SESSIONS : null,
      d: Number.isFinite(left) ? `${left} sessions to the nearer expiry (need ${ENTRY_SESSIONS}); exit before tender` +
        (legs.some((l) => l.front_month === false) ? ". Priced in the next contract: the front month is too close to expiry." : "") : "no contract with enough runway" },
    { k: "Margin", ok: marginOk, d: r.action ? `${inr(r.marginal_margin)} if added · ${pct(r.utilisation_if_added, 1)} of capital · buffer ${r.buffer_state ?? "–"}` +
        (otherReason ? "" : ` · ${r.action}${r.reason ? ", " + r.reason : ""}`) : "not priced" },
  ];
}

function Decision({ p, failed }: { p: PD; failed: string[] }) {
  if (p.status === "ALERT" && p.trade)
    return (
      <div className="decision">
        <div className="h c-good">Alert · suggested pair trade</div>
        {p.trade.legs.map((l) => (
          <div className="leg" key={l.symbol}><span>{l.side === "SELL" ? "Sell" : "Buy"} {l.lots ?? "–"} × <b>{l.symbol}</b> {fexp(l.expiry)}</span>
            <b>{fin(l.fine_g) ? `${num(l.fine_g, 1)} g pure` : ""}</b></div>))}
        <div className="sub">Margin about {inr(p.trade.margin)} · exit when |z| falls to 0.5, at |z| 4.5, or 4 sessions before the nearer expiry</div>
      </div>);
  if (p.status === "NO_TRADE")
    return (
      <div className="decision">
        <div className="h c-warn">No trade · {failed.join(", ") || "gate"}</div><div>{p.reason}</div>
        {p.trade && <div className="sub">Would have been: {p.trade.legs.map((l) => `${l.side.toLowerCase()} ${l.symbol}`).join(", ")}</div>}
      </div>);
  return (
    <div className="decision">
      <div className="h" style={{ color: "var(--ink-2)" }}>{p.status === "WARMUP" ? "Warming up" : "Quiet"}</div>
      <div>{p.reason}. Nothing to do; the pair is inside its normal range.</div>
    </div>);
}

export default function PairDetail() {
  const { id = "" } = useParams();
  const q = usePair(id);
  const risk = useRisk();
  const ov = useOverview();
  const p = usePalette();
  return (
    <Loaded q={q}>{(d) => {
      const r: Risk = { ...(d.risk ?? {}), ...(risk.data?.find((x) => x.pair === d.pair) ?? {}) };
      const ze = d.z_entry;
      const hs = d.history.filter((x) => fin(x.spread_bp) && fin(x.centre) && fin(x.scale));
      const data = hs.map((x) => {
        const extra = (x.spread_bp as number) - (x.centre as number), w = ze * (x.scale as number);
        const hot = fin(x.z) && Math.abs(x.z) >= ze;
        return { t: ms(x.date), extra, band: [-w, w] as [number, number], z: x.z,
                 passed: hot && x.signal_ok ? extra : null, refused: hot && !x.signal_ok ? extra : null };
      });
      const cs = checks(d, r, ov.data?.account);
      const failed = cs.filter((c) => c.ok === false).map((c) => c.k);
      const nAbs = Math.abs(d.centre_bp ?? 0), xAbs = Math.abs(d.dislocation_bp ?? 0), tot = nAbs + xAbs || 1;
      const hot = fin(d.z) && Math.abs(d.z) >= ze;
      return (<>
        <div className="row">
          <Link className="btn" to="/pairs">← Pairs</Link>
          <h2 style={{ margin: 0, fontSize: 25, letterSpacing: "-.02em" }}>{vs(d.pair)}</h2>
          <span className="sub" style={{ fontSize: 14 }}>{(d.legs ?? []).map((l) => `${l.symbol} ${fexp(l.expiry_date)}`).join(" vs ")} · expiry gap {d.expiry_gap_days ?? "–"} days</span>
          <Chip s={d.status} />
        </div>
        <div className="grid2">
          <Card title="Extra gap vs its own history" note={`last ${hs.length} sessions · band uses data to the previous session only`}>
            <Legend items={[{ name: "extra gap", color: p.s1 }, { name: `entry band, ±${ze} × trailing scale`, color: p.band, kind: "box" },
              { name: "unusual, gate refused", color: p.gold, kind: "dot" }, { name: "unusual, all gates passed", color: p.good, kind: "dot" }]} />
            {data.length > 2 ? (
              <Box height={300} label="Extra gap over time against the entry band">
                <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
                  <CartesianGrid stroke={p.grid} vertical={false} />
                  <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmon} minTickGap={40} {...axis(p)} />
                  <YAxis tickFormatter={(v) => `${v} bp`} width={62} {...axis(p)} />
                  <ReferenceLine y={0} stroke={p.grid} />
                  <Area dataKey="band" stroke="none" fill={p.band} fillOpacity={0.9} isAnimationActive={false} />
                  <Line dataKey="extra" stroke={p.s1} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Scatter dataKey="refused" fill={p.gold} isAnimationActive={false} />
                  <Scatter dataKey="passed" fill={p.good} isAnimationActive={false} />
                  <SoftTooltip head={(t) => fdate(new Date(t as number).toISOString())}
                    rows={(pl) => { const row = pl?.[0]?.payload; return row ? [{ label: "extra gap", value: bp(row.extra), color: p.s1 }, { label: "z", value: num(row.z) }] : []; }} />
                </ComposedChart>
              </Box>) : <p className="empty">Not enough history yet.</p>}
            <p>Gap today <b>{bp(d.spread_bp)}</b> = normal <b>{bp(d.centre_bp)}</b> + extra <b>{bp(d.dislocation_bp)}</b></p>
            <div className="decomp" aria-label="Gap split into normal and extra">
              {/* Widths follow the split, but neither label may be cut: each part is at least as wide as its text. */}
              <div className="n" style={{ flex: `${Math.max(nAbs / tot, 0.05)} 0 auto` }}>normal {bp(d.centre_bp)}</div>
              <div className={"x" + (hot ? " hot" : "")} style={{ flex: `${Math.max(xAbs / tot, 0.05)} 0 auto` }}>extra {bp(d.dislocation_bp)} · z = {num(d.z)}</div>
            </div>
            <p className="note">{d.maturity_matched ? "Same expiry window, so the normal gap is mostly each coin's premium."
              : "Different expiry windows, so the normal gap includes carry between the two expiries as well as each coin's premium."}</p>
          </Card>
          <Card title="Checks">
            <div className="checks">{cs.map((c) => (
              <div key={c.k} className={"check" + (c.ok === false ? " fail" : "")}>
                <span className={"dot " + (c.ok === true ? "good" : c.ok === false ? "bad" : "off")} aria-hidden="true">{c.ok === true ? "✓" : c.ok === false ? "✕" : "–"}</span>
                <b>{c.k} <span className="sub">{c.ok === true ? "pass" : c.ok === false ? "fail" : "n/a"}</span></b>
                <div className="d">{c.d}</div>
              </div>))}
            </div>
            <Decision p={d} failed={failed} />
          </Card>
        </div>
        <Card title="Telegram message" note="what the bot sends to the trader">
          {d.telegram ? <div className="well"><pre className="tg">{d.telegram}</pre></div>
            : <p className="note">No message for this pair today. The bot only writes when a pair becomes an alert or an alert clears, so an unchanged state never re-sends.</p>}
        </Card>
      </>);
    }}</Loaded>
  );
}
