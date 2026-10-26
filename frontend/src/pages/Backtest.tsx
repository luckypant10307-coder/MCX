import { Area, AreaChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts";
import { useBacktest, useResearch } from "../api";
import { bp, fdate, fmon, inr, lakh, ms, num, pct, vs } from "../format";
import { usePalette } from "../theme";
import { Box, SoftTooltip, axis } from "../components/charts";
import { Card, Kpi, Loaded } from "../components/ui";

export default function Backtest() {
  const bt = useBacktest();
  const rs = useResearch();
  const p = usePalette();
  return (
    <Loaded q={bt}>{(b) => {
      const perf = b.performance;
      const eq = (rs.data?.equity ?? []).map((x) => ({ t: ms(x.date), v: x.equity }));
      return (<>
        <div className="kpis">
          <Kpi label="Trades" value={perf.n_trades ?? 0} sub={`avg ${num(perf.mean_sessions_held, 1)} sessions held`} />
          <Kpi label="Net P&L" value={lakh(perf.net_pnl)} tone={(perf.net_pnl ?? 0) < 0 ? "bad" : "good"} sub={`gross ${lakh(perf.gross_pnl)} · costs ${lakh(perf.total_cost)}`} />
          <Kpi label="Sharpe" value={num(perf.sharpe)} sub="annualised" />
          <Kpi label="Max drawdown" value={lakh(perf.max_drawdown_inr)} tone="bad" sub={`peak margin ${lakh(perf.peak_margin)}`} />
        </div>
        {eq.length > 0 && (
          <Card title="Cumulative net P&L" note={`after all costs, marked to market daily · ${fdate(rs.data?.window.start)} to ${fdate(rs.data?.window.end)}`}>
            <Box height={290} label="Cumulative net profit and loss">
              <AreaChart data={eq} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
                <defs><linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={p.s1} stopOpacity={0.22} /><stop offset="100%" stopColor={p.s1} stopOpacity={0.02} />
                </linearGradient></defs>
                <CartesianGrid stroke={p.grid} vertical={false} />
                <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmon} minTickGap={40} {...axis(p)} />
                <YAxis tickFormatter={(v) => `₹${(v / 1e5).toFixed(1)}L`} width={64} {...axis(p)} />
                <ReferenceLine y={0} stroke={p.muted} />
                <Area dataKey="v" stroke={p.s1} strokeWidth={2} fill="url(#eqfill)" dot={false} isAnimationActive={false} />
                <SoftTooltip head={(t) => fdate(new Date(t as number).toISOString())} rows={(pl) => [{ label: "net P&L", value: lakh(pl?.[0]?.value), color: p.s1 }]} />
              </AreaChart>
            </Box>
          </Card>)}
        {rs.data?.by_pair && (
          <Card title="By pair"><div className="tw"><table>
            <thead><tr><th>Pair</th><th className="num">Trades</th><th className="num">Gross / trade</th><th className="num">Net / trade</th><th className="num">Net hit rate</th><th className="num">Net P&amp;L</th></tr></thead>
            <tbody>{rs.data.by_pair.map((r) => (
              <tr key={r.pair}><td><b>{vs(r.pair)}</b></td><td className="num">{r.n}</td><td className="num">{bp(r.gross_bp)}</td>
                <td className={"num" + (r.net_bp < 0 ? " c-bad" : "")}>{bp(r.net_bp)}</td><td className="num">{pct(r.hit_rate_net)}</td><td className="num">{inr(r.net_pnl)}</td></tr>))}
            </tbody></table></div></Card>)}
        <Card title="Trades" note="most recent first">
          <div className="tw"><table>
            <thead><tr><th>Pair</th><th>Entry</th><th>Exit</th><th className="num">Lots</th><th className="num">Entry z</th><th className="num">Gross</th><th className="num">Net</th><th>Exit</th></tr></thead>
            <tbody>{b.trades.length ? b.trades.map((t, i) => (
              <tr key={i}><td className="nw"><b>{vs(t.pair)}</b></td><td className="nw">{fdate(t.entry_date)}</td><td className="nw">{fdate(t.exit_date)}</td>
                <td className="num">{t.lots_a} / {t.lots_b}</td><td className="num">{num(t.entry_z)}</td><td className="num">{bp(t.gross_bp)}</td>
                <td className={"num" + (t.net_bp < 0 ? " c-bad" : "")}>{bp(t.net_bp)}</td><td className="sub">{String(t.exit_reason).toUpperCase()}</td></tr>))
              : <tr><td colSpan={8} className="empty">No trades.</td></tr>}</tbody>
          </table></div>
          {perf.tstat_caveat && <p className="note">{perf.tstat_caveat}</p>}
        </Card>
      </>);
    }}</Loaded>
  );
}
