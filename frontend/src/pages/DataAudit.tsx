import { useDataQuality, useHealth, useResearch } from "../api";
import { fdate, int, ist } from "../format";
import { Card, Chip, Kpi, Loaded } from "../components/ui";

export default function DataAudit() {
  const q = useDataQuality();
  const h = useHealth();
  const r = useResearch();
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
