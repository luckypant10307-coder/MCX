import { useMemo, useState } from "react";
import { useSettings, type Parameter, type Settings as S } from "../api";
import { fin, inr, ms, pct } from "../format";
import { Card, Loaded } from "../components/ui";

const fmt = (p: Parameter) => (p.unit === "₹" ? inr(p.value) : `${p.value}${p.unit ? " " + p.unit : ""}`);

function WhatIf({ s }: { s: S }) {
  const [z, setZ] = useState(s.z_entry);
  const step = (d: number) => setZ((v) => Math.max(0.5, Math.min(5, Math.round((v + d) * 10) / 10)));
  const stats = useMemo(() => {
    const hist = Object.values(s.pair_history);
    const last = Math.max(...hist.map((h) => (h.length ? ms(h[h.length - 1].date) : 0)));
    let today = 0, total = 0, n = 0;
    for (const h of hist) for (const r of h) {
      if (!fin(r.z)) continue;
      n++;
      if (Math.abs(r.z) >= z) { total++; if (ms(r.date) === last) today++; }
    }
    return { today, total, n, pairs: hist.length, sessions: Math.max(...hist.map((h) => h.length)) };
  }, [s, z]);
  return (
    <Card title="What if the threshold changed?" note="counts only">
      <div className="stepper">
        <span style={{ flex: 1 }}><b>Unusual if |z| ≥</b><div className="sub">this run used {s.z_entry}</div></span>
        <button onClick={() => step(-0.1)} aria-label="Lower threshold">−</button>
        <output aria-live="polite">{z.toFixed(1)}</output>
        <button onClick={() => step(0.1)} aria-label="Raise threshold">+</button>
      </div>
      <table>
        <thead><tr><th /><th className="num">Today</th><th className="num">Last {stats.sessions} sessions</th></tr></thead>
        <tbody>
          <tr><td>Unusual pair-sessions</td><td className="num"><b>{stats.today}</b></td><td className="num"><b>{stats.total}</b></td></tr>
          <tr><td>Share of all checks</td><td className="num">{pct(stats.today / (stats.pairs || 1))}</td><td className="num">{pct(stats.total / (stats.n || 1), 1)}</td></tr>
        </tbody>
      </table>
      <p className="note">This only counts pair-sessions whose z crosses the threshold, from the stored history. The other gates, the risk board and the backtest need a full re-run to change.</p>
    </Card>
  );
}

export default function Settings() {
  const q = useSettings();
  return (
    <Loaded q={q}>{(s) => {
      const groups = [...new Set(s.parameters.map((p) => p.group))];
      return (
        <div className="grid2">
          <Card title="Engine parameters" note="used by this run · read-only">
            {groups.map((g) => (
              <div key={g}>
                <div className="sub" style={{ fontWeight: 800, letterSpacing: ".07em", textTransform: "uppercase", margin: "6px 0 2px" }}>{g}</div>
                <table className="params"><tbody>{s.parameters.filter((p) => p.group === g).map((p) => (
                  <tr key={p.name}><td><b>{p.name}</b></td><td className="num"><b>{fmt(p)}</b></td><td className="sub">{p.what}</td></tr>))}
                </tbody></table>
              </div>))}
            <p className="note">To change one, re-run the nightly job with the new value, for example <span className="mono">python scripts/nightly.py --source csv --z-entry 2.5</span>. The engine is rule-based; nothing here is learned or tuned by AI.</p>
          </Card>
          <WhatIf s={s} />
        </div>
      );
    }}</Loaded>
  );
}
