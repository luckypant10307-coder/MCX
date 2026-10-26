import { Link } from "react-router-dom";
import { useDecisions, useOverview, type Decision } from "../api";
import { bp, fdate, int, num, pairUrl, pct, vs } from "../format";
import { Chip, Kpi, Loaded } from "../components/ui";

// Short codes like a trader's log; the full sentence is in the row's title.
const REASON: Record<string, string> = {
  "not enough history": "HISTORY", "spread not stationary": "NOT MEAN-REVERTING",
  "half-life too long": "HALF-LIFE", "dislocation too small": "SIZE",
};
const code = (r: Decision) =>
  r.status === "TRADED" ? "ENTERED NEXT SESSION" : r.status === "SIGNAL" ? "NOT ENTERED · BOOK FULL OR SIZE"
    : r.reason.split(", ").map((x) => REASON[x] ?? x.toUpperCase()).join(", ");

export default function Decisions() {
  const dq = useDecisions();
  const ov = useOverview();
  return (
    <Loaded q={dq}>{(d) => {
      const win = ov.data?.meta.decision_window_sessions ?? 120, n = ov.data?.summary.pairs ?? 6, checks = win * n;
      return (<>
        <div className="kpis">
          <Kpi label="Checks run" value={int(checks)} sub={`${n} pairs × ${win} sessions`} />
          <Kpi label="Quiet" value={pct(1 - d.total / (checks || 1), 1)} sub="nothing unusual" />
          <Kpi label="Unusual, no trade" value={d.counts.NO_TRADE ?? 0} tone="warn" sub="each with the gate that refused" />
          <Kpi label="Passed every gate" value={(d.counts.TRADED ?? 0) + (d.counts.SIGNAL ?? 0)} tone="good" sub={`${d.counts.TRADED ?? 0} entered by the backtest next session`} />
        </div>
        <section className="card">
          <h2>Decision log<small>every unusual pair-session, with the check that decided it</small></h2>
          <div className="tw"><table>
            <thead><tr><th>Date</th><th>Pair</th><th className="num">z</th><th className="num">Gap</th><th className="num">Extra</th><th>Decision</th><th>Reason</th><th /></tr></thead>
            <tbody>{d.items.length ? d.items.map((r, i) => (
              <tr key={i} title={r.reason}>
                <td className="nw">{fdate(r.date)}</td><td className="nw"><b>{vs(r.pair)}</b></td><td className="num">{num(r.z)}</td>
                <td className="num">{bp(r.spread_bp)}</td><td className="num"><b>{bp(r.dislocation_bp)}</b></td>
                <td><Chip s={r.status} /></td><td className="sub">{code(r)}</td>
                <td className="num"><Link className="btn" to={"/pairs/" + pairUrl(r.pair)}>View pair</Link></td>
              </tr>)) : <tr><td colSpan={8} className="empty">Nothing unusual in the window.</td></tr>}
            </tbody>
          </table></div>
        </section>
      </>);
    }}</Loaded>
  );
}
