import { Link } from "react-router-dom";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { useDataQuality, useOverview } from "../api";
import { compact, fdate, fexp, fmon, int, ms } from "../format";
import { usePalette } from "../theme";
import { Box, SoftTooltip, axis } from "../components/charts";
import { Card, Chip, Kpi, Loaded } from "../components/ui";

export default function Overview() {
  const ov = useOverview();
  const dq = useDataQuality();
  const p = usePalette();
  return (
    <Loaded q={ov}>{(o) => {
      const s = o.summary;
      const last = dq.data?.archive?.rows?.[0];
      const dateOk = !last || last.status === "LOADED";
      const g = o.contracts.map((c) => c.inr_per_g_fine);
      const spreadBp = g.length ? (Math.max(...g) / Math.min(...g) - 1) * 1e4 : null;
      const gold = o.gold_level.map((r) => ({ t: ms(r.date), v: r.inr_per_10g_fine / 10 }));
      return (<>
        <div className="kpis">
          <Kpi label="Bhavcopy" value={fdate(o.meta.asof)}
               sub={<><Chip s={dateOk ? "ok" : "bad"} label={dateOk ? "date check ok" : "date mismatch"} />
                     {last && <span>{last.gold_rows} gold rows{dq.data?.archive ? `, ${dq.data.archive.rejected} rejected` : ""}</span>}</>} />
          <Kpi label="Alerts today" value={s.ALERT} tone="good" sub={s.ALERT ? "sent to Telegram on change" : "all gates and risk checks must pass"} />
          <Kpi label="No trade today" value={s.NO_TRADE} tone="warn" sub="unusual, logged with the gate that refused" />
          <Kpi label="Quiet pairs" value={s.QUIET} sub={s.WARMUP ? `+${s.WARMUP} warming up` : "nothing unusual"} />
        </div>

        <div className="grid2 wide">
          <Card title="Contracts today" note="front month, converted to ₹ per gram of pure gold">
            <div className="tw"><table>
              <thead><tr><th>Contract</th><th>Expiry</th><th className="num">Close (₹)</th><th className="num">Quote basis</th>
                <th className="num">₹ / g pure</th><th>Liquidity</th></tr></thead>
              <tbody>{o.contracts.map((c) => (
                <tr key={c.symbol} title={`₹${int(c.close)} ÷ (${c.quote_base_g} g × ${Math.round(c.purity * 1000)}/1000) = ₹${int(c.inr_per_g_fine)} per g`}>
                  <td><b>{c.symbol}</b></td><td className="nw">{fexp(c.expiry_date)}</td><td className="num">{int(c.close)}</td>
                  <td className="num">{c.quote_base_g} g · {Math.round(c.purity * 1000)}</td>
                  <td className="num"><b>{int(c.inr_per_g_fine)}</b></td>
                  <td className="nw">{c.liquidity && <Chip s={c.liquidity} />}<div className="sub" style={{ marginTop: 4 }}>{compact(c.avg_volume_20)} lots/day</div></td>
                </tr>))}
              </tbody>
            </table></div>
            <p className="note">Quote basis is the grams the price refers to and the purity (995 or 999 parts per thousand). Same metal, four quote formats. After conversion they sit within {spreadBp == null ? "–" : spreadBp.toFixed(0)} bp of each other.
              Liquidity is average lots traded per day over 20 sessions (K = thousand, L = lakh) against each contract's minimum: green ≥ 2×, yellow ≥ 1×.</p>
          </Card>

          <Card title="Gold, ₹ per gram of pure gold" note={`fitted level · last ${gold.length} sessions`}>
            <Box height={280} label="Fitted gold level in rupees per gram of fine gold">
              <AreaChart data={gold} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
                <defs><linearGradient id="goldfill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={p.gold} stopOpacity={0.28} /><stop offset="100%" stopColor={p.gold} stopOpacity={0.02} />
                </linearGradient></defs>
                <CartesianGrid stroke={p.grid} vertical={false} />
                <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmon} {...axis(p)} minTickGap={40} />
                <YAxis domain={["auto", "auto"]} tickFormatter={(v) => int(v)} width={58} {...axis(p)} />
                <SoftTooltip head={(t) => fdate(new Date(t as number).toISOString())} rows={(pl) => [{ label: "₹ per g pure", value: "₹" + int(pl?.[0]?.value), color: p.gold }]} />
                <Area type="monotone" dataKey="v" stroke={p.gold} strokeWidth={2} fill="url(#goldfill)" dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
              </AreaChart>
            </Box>
          </Card>
        </div>

        <section className="card">
          <div className="verdict">
            <div className="t">
              <span className={"dot " + (o.verdict.edge_survives_costs ? "good" : "bad")} aria-hidden="true">{o.verdict.edge_survives_costs ? "✓" : "✕"}</span>
              <div><b>Research verdict:</b> {o.verdict.headline}</div>
            </div>
            <Link className="btn" to="/research">See the evidence →</Link>
          </div>
        </section>
      </>);
    }}</Loaded>
  );
}
