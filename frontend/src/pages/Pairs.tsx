import { useNavigate } from "react-router-dom";
import { useOverview, usePairs } from "../api";
import { bp, fdate, fexp, fin, num, pairUrl, vs } from "../format";
import { Chip, Loaded } from "../components/ui";

export function ZTrack({ z, entry }: { z: number | null; entry: number }) {
  const pos = (v: number) => Math.max(0, Math.min(100, ((v + 4) / 8) * 100));
  return (
    <div className="ztrack" aria-hidden="true">
      <div className="zone" style={{ left: 4, width: `calc(${pos(-entry)}% - 4px)` }} />
      <div className="zone" style={{ left: `${pos(entry)}%`, right: 4 }} />
      <div className="mid" />
      {fin(z) && <div className={"knob" + (Math.abs(z) >= entry ? " hot" : "")} style={{ left: `${pos(z)}%` }} />}
    </div>
  );
}

export default function Pairs() {
  const pairs = usePairs();
  const ov = useOverview();
  const go = useNavigate();
  return (
    <Loaded q={pairs}>{(ps) => {
      const ze = ov.data?.meta.z_entry ?? 2;
      const s = ov.data?.summary;
      return (<>
        {s && <section className="card"><p className="lede"><b>{fdate(ov.data?.meta.asof)}:</b> {s.ALERT} alert, {s.NO_TRADE} no-trade, {s.QUIET} quiet
          {s.WARMUP ? `, ${s.WARMUP} warming up` : ""}. The normal gap is each pair's own trailing centre, which already contains carry between expiries
          and each coin's premium; only the extra is scored.</p></section>}
        <section className="card"><div className="tw"><table>
          <thead><tr><th>Pair</th><th>Contracts</th><th className="num">Gap</th><th className="num">Normal</th><th className="num">Extra</th>
            <th>z, −4 to +4</th><th className="num">z (t−1)</th><th>Decision</th><th>Reason</th></tr></thead>
          <tbody>{ps.map((p) => (
            <tr key={p.pair} className="go" tabIndex={0} onClick={() => go("/pairs/" + pairUrl(p.pair))}
                onKeyDown={(e) => e.key === "Enter" && go("/pairs/" + pairUrl(p.pair))}>
              <td><b className="nw">{vs(p.pair)}</b><div className="sub">{p.maturity_matched ? "same expiry window" : "carry-bridged"}</div></td>
              <td className="sub nw">{(p.legs ?? []).map((l) => fexp(l.expiry_date)).join(" / ")}</td>
              <td className="num">{bp(p.spread_bp)}</td><td className="num">{bp(p.centre_bp)}</td><td className="num"><b>{bp(p.dislocation_bp)}</b></td>
              <td><ZTrack z={p.z} entry={ze} /></td><td className="num"><b>{num(p.z)}</b></td>
              <td><Chip s={p.status} /></td><td className="sub">{p.reason}</td>
            </tr>))}
          </tbody>
        </table></div></section>
      </>);
    }}</Loaded>
  );
}
