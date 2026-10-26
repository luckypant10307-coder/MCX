import { Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, LabelList, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts";
import { useResearch, type Research as R } from "../api";
import { bp, fdate, fmon, inr, int, ms, num, pct } from "../format";
import { usePalette, type Palette } from "../theme";
import { Box, SoftTooltip, axis } from "../components/charts";
import { Card, Kpi, Legend, Loaded, TableView } from "../components/ui";

const BANDS: [string, string, keyof Palette][] = [["|z| 0-1", "|z| below 1", "o1"], ["|z| 1-2", "|z| 1 to 2", "o2"], ["|z| 2-3", "|z| 2 to 3", "o3"], ["|z| 3-inf", "|z| 3 and above", "o4"]];
const PREM: [string, keyof Palette][] = [["GOLDGUINEA", "s1"], ["GOLDPETAL", "s2"], ["GOLDTEN", "s3"]];

function EventStudy({ r, p }: { r: R; p: Palette }) {
  const es = r.event_study!;
  const hz = [...new Set(es.rows.map((x) => x.horizon))].sort((a, b) => a - b);
  const data = hz.map((h) => {
    const row: Record<string, number | [number, number]> = { h };
    for (const [k] of BANDS) { const e = es.rows.find((x) => x.z_band === k && x.horizon === h); if (e) row[k] = e.mean_conv_bp; }
    const top = es.rows.find((x) => x.z_band === "|z| 3-inf" && x.horizon === h);
    if (top) row.ci = [top.boot_ci_lo, top.boot_ci_hi];
    return row;
  });
  const clears = es.rows.filter((x) => x.ci_clears_costs);
  return (
    <Card title="1 · Wide spreads do close, and wider ones close further">
      <p className="note">Average convergence after a dislocation, by |z| band and sessions waited. Every pair-day counts, tradable or not.
        Shaded: 95% bootstrap interval for |z| ≥ 3. Dashed: round-trip cost at a {es.half_spread_bp} bp spread.</p>
      <Legend items={[...BANDS.map(([, name, c]) => ({ name, color: p[c] })), { name: "round-trip cost", color: p["ink-2"], kind: "dash" as const }]} />
      <Box height={320} label="Mean convergence by horizon for four |z| bands with the cost line">
        <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 6, left: 4 }}>
          <CartesianGrid stroke={p.grid} vertical={false} />
          <XAxis dataKey="h" type="number" ticks={hz} domain={[hz[0], hz[hz.length - 1]]} {...axis(p)}
                 label={{ value: "sessions held", position: "insideBottom", offset: -2, fill: p.muted, fontSize: 11.5 }} height={36} />
          <YAxis tickFormatter={(v) => `${v} bp`} width={62} {...axis(p)} />
          <Area dataKey="ci" stroke="none" fill={p.band} fillOpacity={0.9} isAnimationActive={false} />
          <ReferenceLine y={es.round_trip_bp} stroke={p["ink-2"]} strokeDasharray="6 4" strokeWidth={1.5}
                         label={{ value: `cost ${es.round_trip_bp.toFixed(0)} bp`, position: "insideTopLeft", fill: p["ink-2"], fontSize: 12 }} />
          {BANDS.map(([k, , c]) => <Line key={k} dataKey={k} stroke={p[c]} strokeWidth={2.2} dot={{ r: 4, strokeWidth: 0, fill: p[c] }} isAnimationActive={false} />)}
          <SoftTooltip head={(h) => `after ${h} session${h === 1 ? "" : "s"}`}
            rows={(pl) => BANDS.map(([k, name, c]) => ({ k, name, c, v: pl?.find((x) => x.dataKey === k)?.value }))
              .filter((x) => x.v != null).map((x) => ({ label: x.name, value: bp(x.v), color: p[x.c] }))} />
        </ComposedChart>
      </Box>
      <p>{clears.length ? `Only ${clears.map((x) => `|z| ≥ 3 held ${x.horizon} sessions`).join(", ")} clears the cost line with 95% confidence.` : "No band clears the cost line with 95% confidence."}
        {" "}The backtest rarely holds that long: {r.exit_reasons?.stop ?? "–"} of {r.performance.n_trades} trades were stopped out first.</p>
      <TableView head={["band", "sessions", "n", "mean bp", "95% CI", "NW t", "net of cost"]}
        rows={es.rows.map((x) => [x.z_band, x.horizon, x.n, num(x.mean_conv_bp, 1), `${num(x.boot_ci_lo, 1)} to ${num(x.boot_ci_hi, 1)}`, num(x.t_newey_west), num(x.net_conv_bp, 1)])} />
    </Card>);
}

function Costs({ r, p }: { r: R; p: Palette }) {
  const c = r.costs, s0 = c.sweep[0];
  const stat = s0.statutory_bp, per = (s0.round_trip_bp - stat) / s0.half_spread_bp_per_leg;
  const data = [0, 1, 2, 3, 4, 5, 6, 8, 10].map((x) => ({ x, cost: stat + per * x }));
  return (
    <Card title="2 · Costs pass the gross edge almost immediately">
      <p className="note">Four bid-ask crossings per trade. Statutory charges alone are {stat.toFixed(1)} bp; the spread itself is not in bhavcopy, so cost is shown as it widens.</p>
      <Legend items={[{ name: "round-trip cost", color: p.s1 }, { name: "gross edge per trade", color: p["ink-2"], kind: "dash" }]} />
      <Box height={260} label="Round-trip cost against assumed half-spread">
        <LineChart data={data} margin={{ top: 10, right: 16, bottom: 6, left: 4 }}>
          <CartesianGrid stroke={p.grid} vertical={false} />
          <XAxis dataKey="x" type="number" domain={[0, 10]} ticks={[0, 2, 4, 6, 8, 10]} tickFormatter={(v) => `${v} bp`} {...axis(p)} height={36}
                 label={{ value: "assumed half-spread per leg", position: "insideBottom", offset: -2, fill: p.muted, fontSize: 11.5 }} />
          <YAxis tickFormatter={(v) => `${v} bp`} width={58} domain={[0, "auto"]} {...axis(p)} />
          <ReferenceLine y={c.gross_bp} stroke={p["ink-2"]} strokeDasharray="6 4" strokeWidth={1.5}
                         label={{ value: `gross ${c.gross_bp.toFixed(1)} bp`, position: "insideTopRight", fill: p["ink-2"], fontSize: 12 }} />
          <Line dataKey="cost" stroke={p.s1} strokeWidth={2.2} dot={false} isAnimationActive={false} />
          <SoftTooltip head={(x) => `half-spread ${x} bp per leg`} rows={(pl) => [{ label: "round trip", value: `${num(pl?.[0]?.value, 1)} bp`, color: p.s1 }]} />
        </LineChart>
      </Box>
      <p>They cross at <b>{num(c.breakeven_half_spread_bp, 1)} bp per leg</b>.</p>
    </Card>);
}

function Premiums({ r, p }: { r: R; p: Palette }) {
  const byDate = new Map<number, Record<string, number>>();
  for (const x of r.premiums ?? []) { const t = ms(x.date); byDate.set(t, { ...(byDate.get(t) ?? { t }), [x.symbol]: x.premium_bp }); }
  const data = [...byDate.values()].sort((a, b) => a.t - b.t);
  return (
    <Card title="3 · Coin premiums drift slowly, and are removed">
      <p className="note">Premium over the GOLDM 995 bar, re-estimated weekly from the previous 120 sessions. It moves over months; left in, it would swamp the 20–70 bp dislocations being traded.</p>
      <Legend items={PREM.map(([s, c]) => ({ name: s, color: p[c] }))} />
      <Box height={260} label="Coin premiums over time">
        <LineChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: 4 }}>
          <CartesianGrid stroke={p.grid} vertical={false} />
          <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmon} minTickGap={40} {...axis(p)} />
          <YAxis tickFormatter={(v) => `${v} bp`} width={62} {...axis(p)} />
          <ReferenceLine y={0} stroke={p.grid} />
          {PREM.map(([s, c]) => <Line key={s} dataKey={s} stroke={p[c]} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />)}
          <SoftTooltip head={(t) => fdate(new Date(t as number).toISOString())}
            rows={(pl) => (pl ?? []).map((x) => ({ label: String(x.dataKey), value: bp(x.value), color: String(x.color) }))} />
        </LineChart>
      </Box>
    </Card>);
}

function Years({ r, p }: { r: R; p: Palette }) {
  const ys = r.by_year ?? [];
  return (
    <Card title="4 · Net per trade, by entry year">
      <p className="note">Only 2022 was positive. GOLDTEN pairs exist only from 2025.</p>
      <Legend items={[{ name: "positive", color: p.pos, kind: "box" }, { name: "negative", color: p.neg, kind: "box" }]} />
      <Box height={240} label="Net basis points per trade by entry year">
        <BarChart data={ys} margin={{ top: 22, right: 8, bottom: 0, left: 4 }}>
          <CartesianGrid stroke={p.grid} vertical={false} />
          <XAxis dataKey="year" {...axis(p)} />
          <YAxis tickFormatter={(v) => `${v} bp`} width={58} {...axis(p)} />
          <ReferenceLine y={0} stroke={p.muted} />
          <Bar dataKey="net_bp" radius={[6, 6, 6, 6]} isAnimationActive={false}>
            {ys.map((y) => <Cell key={y.year} fill={y.net_bp >= 0 ? p.pos : p.neg} />)}
            <LabelList dataKey="net_bp" position="top" formatter={(v: number) => num(v, 1)} fill={p["ink-2"]} fontSize={12} />
          </Bar>
          <SoftTooltip head={(y) => String(y)} rows={(pl) => { const y = pl?.[0]?.payload; return y ? [
            { label: "net per trade", value: bp(y.net_bp) }, { label: "gross", value: bp(y.gross_bp) },
            { label: "trades", value: String(y.n) }, { label: "net P&L", value: inr(y.net_pnl) }] : []; }} />
        </BarChart>
      </Box>
      <TableView head={["year", "trades", "gross bp", "net bp", "net P&L"]} rows={ys.map((y) => [y.year, y.n, num(y.gross_bp, 1), num(y.net_bp, 1), inr(y.net_pnl)])} />
    </Card>);
}

function Exits({ r, p }: { r: R; p: Palette }) {
  const ex = r.exit_reasons ?? {}, tot = Object.values(ex).reduce((a, b) => a + b, 0) || 1;
  const parts: [string, string, string][] = ([["stop", "stopped out", p.s2], ["converged", "converged", p.s1], ["calendar", "expiry runway", p.s3], ["max_holding", "held too long", p.o2]] as [string, string, string][]).filter(([k]) => ex[k]);
  return (
    <Card title="Why trades end">
      <p className="note">A spread that keeps widening hits the stop; contracts nearing expiry force the rest out before convergence.</p>
      <Legend items={parts.map(([, name, color]) => ({ name, color, kind: "box" as const }))} />
      <div className="prop">{parts.map(([k, name, color]) => (
        <div key={k} style={{ flex: ex[k], background: color }} title={`${name}: ${ex[k]} trades (${pct(ex[k] / tot)})`}>{ex[k] / tot > 0.12 ? ex[k] : ""}</div>))}</div>
      <table><tbody>{parts.map(([k, name]) => <tr key={k}><td>{name}</td><td className="num">{ex[k]}</td><td className="num">{pct(ex[k] / tot)}</td></tr>)}
        <tr><td>average holding</td><td className="num" colSpan={2}>{num(r.performance.mean_sessions_held, 1)} sessions</td></tr></tbody></table>
    </Card>);
}

export default function Research() {
  const q = useResearch();
  const p = usePalette();
  return (
    <Loaded q={q}>{(r) => {
      const c = r.costs, ok = r.verdict.edge_survives_costs;
      const cost = c.gross_bp - c.net_bp;
      return (<>
        <section className="card">
          <div className="verdict"><div className="t">
            <span className={"dot " + (ok ? "good" : "bad")} aria-hidden="true">{ok ? "✓" : "✕"}</span>
            <div><b>{ok ? "An edge survives costs." : "The dislocation is real. The edge does not survive costs."}</b> {r.verdict.headline} · {int(r.window.sessions)} sessions, {fdate(r.window.start)} to {fdate(r.window.end)}.</div>
          </div></div>
        </section>
        <div className="kpis">
          <Kpi label="Gross per trade" value={bp(c.gross_bp)} sub="what convergence paid" />
          <Kpi label="Costs per trade" value={`${cost.toFixed(1)} bp`} sub="fees, taxes, spread, slippage" />
          <Kpi label="Net per trade" value={bp(c.net_bp)} tone={c.net_bp < 0 ? "bad" : "good"} sub={`${r.performance.n_trades} trades · hit rate ${pct(r.performance.hit_rate_net)}`} />
          <Kpi label="Breakeven spread" value={`${num(c.breakeven_half_spread_bp, 1)} bp`} sub="per leg, per crossing" />
        </div>
        {r.event_study && <EventStudy r={r} p={p} />}
        <div className="grid2 even"><Costs r={r} p={p} /><Premiums r={r} p={p} /></div>
        <div className="grid2 even"><Years r={r} p={p} /><Exits r={r} p={p} /></div>
        <Card title="How the engine is kept honest" note="run on every push">
          <div className="checks">{[["Null world", "On synthetic data with no relative value, the strategy must show no edge."],
            ["Known truth", "A synthetic world with an injected premium and dislocation, which the model must recover."],
            ["No look-ahead", "Cutting off future data must not change any past signal."],
            ["Next-session fills", "A signal from today's settlement is filled tomorrow, never at the price that produced it."]].map(([k, d]) => (
              <div className="check" key={k}><span className="dot good" aria-hidden="true">✓</span><b>{k}</b><div className="d">{d}</div></div>))}</div>
        </Card>
      </>);
    }}</Loaded>
  );
}
