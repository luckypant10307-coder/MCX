import { useAnomalies, useDataQuality, useHealth, useResearch, type Anomalies } from "../api";
import { fdate, fexp, fin, int, ist, num, pct } from "../format";
import { Card, Chip, Kpi, Loaded } from "../components/ui";

// What each detector feature means to a person reading the queue.
const FEATURE: Record<string, string> = {
  ret_in_ticks: "price jump, measured in ticks", ret_vs_peers_bp: "moved unlike the other contracts",
  peer_dev_bp: "price far from the other contracts", log_ret: "unusually large daily move",
  range_frac: "unusually wide high–low range", close_in_range: "close at an odd point in the day's range",
  d_log_volume: "volume jumped or collapsed", d_log_oi: "open interest jumped or collapsed", resid_bp: "large model residual",
};

function AnomalyReview({ a }: { a: Anomalies }) {
  if (!a.available) return <Card title="Anomaly review"><p className="note">Not available: {a.reason}.</p></Card>;
  const top = a.top ?? [];
  return (
    <Card title="Anomaly review queue" note="Isolation Forest · for a person to look at · never blocks data or trades">
      <div className="row" style={{ gap: 28 }}>
        <div><div className="sub">rows scored</div><b style={{ fontSize: 22 }}>{int(a.rows_scored)}</b></div>
        <div><div className="sub">flagged</div><b style={{ fontSize: 22 }}>{int(a.flagged)}</b> <span className="sub">{pct((a.flagged ?? 0) / (a.rows_scored || 1), 1)}</span></div>
        <div><div className="sub">missed by the rule checks</div><b className="c-warn" style={{ fontSize: 22 }}>{int(a.flagged_new)}</b></div>
        <div><div className="sub">already flagged by a rule</div><b style={{ fontSize: 22 }}>{int(a.flagged_known)}</b></div>
      </div>
      <div className="row" style={{ gap: 10 }}>
        {(a.by_feature ?? []).map((f) => <span key={f.top_feature} className="chip QUIET">{FEATURE[f.top_feature] ?? f.top_feature} · {f.n}</span>)}
      </div>
      <div className="tw"><table>
        <thead><tr><th>Date</th><th>Contract</th><th>What looks odd</th><th className="num">Day move</th><th className="num">vs peers</th><th className="num">Volume change</th><th className="num">Score</th></tr></thead>
        <tbody>{top.map((r) => (
          <tr key={r.contract_id + r.date}>
            <td className="nw">{fdate(r.date)}</td>
            <td className="nw"><b>{r.symbol}</b> <span className="sub">{fexp(r.contract_id.split("|")[1])}</span></td>
            <td>{FEATURE[r.top_feature] ?? r.top_feature}<div className="sub">{num(r.top_z, 1)} standard deviations from normal</div></td>
            <td className="num">{fin(r.log_ret) ? pct(Math.expm1(r.log_ret), 1) : "–"}</td>
            <td className="num">{fin(r.peer_dev_bp) ? `${num(r.peer_dev_bp, 0)} bp` : "–"}</td>
            <td className="num">{fin(r.d_log_volume) ? pct(Math.expm1(r.d_log_volume), 0) : "–"}</td>
            <td className="num">{num(r.score, 3)}</td>
          </tr>))}
        </tbody>
      </table></div>
      <p className="note">Showing the {top.length} most unusual rows the rule checks did not already catch, most unusual first (lower score = stranger).
        A flag is a question, not a verdict: a genuine market shock looks exactly like a bad print until someone checks it. The detector is fit on the whole sample,
        which is fine for an audit but would be look-ahead in a backtest, so it never filters anything.</p>
    </Card>
  );
}

export default function DataAudit() {
  const q = useDataQuality();
  const h = useHealth();
  const r = useResearch();
  const an = useAnomalies();
  return (
    <Loaded q={q}>{(d) => {
      const a = d.archive;
      const flags = Object.entries(d.flags).filter(([k]) => k !== "symbol_unknown_or_filtered");
      const ten = d.coverage.find((c) => c.symbol === "GOLDTEN");
      return (<>
        <div className="kpis">
          <Kpi label="Files archived" value={a ? int(a.files) : "–"} sub="raw bytes kept, never edited" />
          <Kpi label="Rows checked" value={int(d.rows_in)} sub={`${int(d.rows_out)} gold futures rows kept`} />
          <Kpi label="Files rejected" value={a ? int(a.rejected) : "–"} tone={a && a.rejected ? "bad" : undefined} sub="date inside ≠ date requested" />
          <Kpi label="Contract specs" value={d.coverage.length} sub="quote basis, purity, expiry window" />
        </div>
        {a ? (
          <Card title="Download log" note={`requested date must equal the date inside the file · latest ${a.rows.length} of ${int(a.files)}`}>
            <div className="tw"><table>
              <thead><tr><th>Requested</th><th>Returned</th><th>Saved (IST)</th><th>File</th><th className="num">Gold rows</th><th>SHA-256</th><th>Status</th></tr></thead>
              <tbody>{a.rows.map((x) => (
                <tr key={x.file}>
                  <td className="nw">{fdate(x.requested)}</td><td className={"nw" + (x.status === "LOADED" ? "" : " c-bad")}><b>{fdate(x.returned)}</b></td>
                  <td className="sub nw">{ist(x.saved_at, true)}</td><td className="mono">{x.file}</td><td className="num">{x.gold_rows}</td>
                  <td className="mono" title={x.sha256}>{x.sha256 ? x.sha256.slice(0, 16) + "…" : "–"}</td><td><Chip s={x.status} /></td>
                </tr>))}
              </tbody>
            </table></div>
          </Card>
        ) : <section className="card"><p className="lede">No download log: this run did not read a directory of bhavcopy files (source: {h.data?.source}).</p></section>}
        {an.data && <AnomalyReview a={an.data} />}
        <div className="grid2">
          <Card title="Coverage" note="per contract family">
            <div className="tw"><table>
              <thead><tr><th>Contract</th><th>First</th><th>Last</th><th className="num">Sessions</th><th className="num">Expiries</th><th className="num">No-trade settles</th></tr></thead>
              <tbody>{d.coverage.map((c) => (
                <tr key={c.symbol}><td><b>{c.symbol}</b></td><td className="nw">{fdate(c.first)}</td><td className="nw">{fdate(c.last)}</td>
                  <td className="num">{int(c.sessions)}</td><td className="num">{c.contracts}</td><td className="num">{int(c.untradeable_rows)}</td></tr>))}
              </tbody>
            </table></div>
            <p className="note">GOLDTEN lists only from {fdate(ten?.first)}, so its pairs rest on less history. No-trade settles are exchange-computed prices on days nothing traded; they are flagged and never filled against.</p>
          </Card>
          <Card title="Flags and provenance">
            <table><tbody>
              {flags.map(([k, v]) => <tr key={k}><td className="mono">{k}</td><td className="num">{int(v)}</td></tr>)}
              <tr><td>source</td><td className="num">{h.data?.source}{h.data?.source === "csv" ? " · MCX bhavcopy via mtf.trading" : ""}</td></tr>
              <tr><td>run</td><td className="num mono">{d.run_id}</td></tr>
              <tr><td>input hash</td><td className="mono" style={{ wordBreak: "break-all", textAlign: "right" }}>{d.data_fingerprint}</td></tr>
              <tr><td>margin parameters</td><td className="num" style={{ whiteSpace: "normal" }}>{r.data?.meta.span_parameters}</td></tr>
            </tbody></table>
          </Card>
        </div>
      </>);
    }}</Loaded>
  );
}
