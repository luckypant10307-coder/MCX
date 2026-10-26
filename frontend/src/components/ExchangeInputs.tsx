import type { Circulars, RiskFiles } from "../api";
import { fdate, ist, pct } from "../format";
import { Card, Chip } from "./ui";

const TOPIC: Record<string, string> = {
  margin: "margin", margin_withdrawal: "margin withdrawn", specification: "contract specification", launch: "new contract",
};

/** The MCXCCL drop folder: what was accepted, what was rejected and why, and coverage. */
export function RiskFilesCard({ r, source }: { r: RiskFiles; source?: string }) {
  return (
    <Card title="MCXCCL risk files" note="drop folder · data/risk/inbox">
      <div className="row" style={{ gap: 28 }}>
        <div><div className="sub">accepted</div><b style={{ fontSize: 22 }} className={r.accepted ? "c-good" : ""}>{r.accepted}</b></div>
        <div><div className="sub">rejected</div><b style={{ fontSize: 22 }} className={r.rejected ? "c-bad" : ""}>{r.rejected}</b></div>
        <div><div className="sub">waiting</div><b style={{ fontSize: 22 }}>{r.waiting}</b></div>
        {r.coverage && <div><div className="sub">sessions covered</div><b style={{ fontSize: 22 }}>{r.coverage.covered}</b> <span className="sub">of {r.coverage.sessions} ({pct(r.coverage.coverage, 1)})</span></div>}
      </div>
      <p className="note">Margin parameters in use: <b>{source ?? "–"}</b>.</p>
      {r.recent.length ? (
        <div className="tw"><table>
          <thead><tr><th>File</th><th>Status</th><th>Business date</th><th>Detail</th><th>Processed</th></tr></thead>
          <tbody>{r.recent.map((d, i) => (
            <tr key={i}><td className="mono">{d.file}</td><td><Chip s={d.status === "ACCEPTED" ? "LOADED" : "RED"} label={d.status.toLowerCase()} /></td>
              <td className="nw">{fdate(d.business_date)}</td>
              <td className="sub">{d.status === "ACCEPTED" ? `${d.kind} · ${(d.commodities ?? []).filter((c) => c.startsWith("GOLD")).join(", ")}` : d.reason}</td>
              <td className="sub nw">{ist(d.at, true)}</td></tr>))}
          </tbody>
        </table></div>
      ) : (
        <div className="well">
          <b>No files yet.</b> MCXCCL blocks automated downloads, so this one step is manual: open the daily margin or SPAN file on
          mcxccl.com in your browser and save it into <span className="mono">data/risk/inbox/</span>. The next nightly run validates it,
          files it as accepted or rejected (with the reason), and uses its scan ranges for that date only. Every other date keeps the
          estimate from volatility, and is labelled as such.
        </div>
      )}
    </Card>
  );
}

/** Circulars found by TinyFish search: links for a person to read, new ones first. */
export function CircularsCard({ c }: { c: Circulars }) {
  if (!c.available)
    return (
      <Card title="MCX circulars" note="TinyFish web search">
        <p className="note">Not running: {c.reason}. Set <span className="mono">TINYFISH_API_KEY</span> in <span className="mono">.env</span> locally,
          or as an environment variable on Render, to watch for new margin and contract circulars.</p>
      </Card>);
  const items = c.items ?? [];
  return (
    <Card title="MCX circulars" note={`TinyFish web search · checked ${ist(c.checked_at, true)} IST · ${c.new ?? 0} new`}>
      {items.length ? (
        <div className="tw"><table>
          <thead><tr><th /><th>Result</th><th>About</th><th>Source</th></tr></thead>
          <tbody>{items.slice(0, 10).map((x) => (
            <tr key={x.url}>
              <td>{x.new ? <Chip s="NO_TRADE" label="new" /> : <Chip s="QUIET" label="seen" />}</td>
              <td><a href={x.url} target="_blank" rel="noopener noreferrer"><b>{x.title ?? x.url}</b></a>
                {x.snippet && <div className="sub clamp2" style={{ maxWidth: "90ch" }}>{x.snippet}</div>}</td>
              <td className="sub">{x.topics.map((t) => TOPIC[t] ?? t).join(", ")}</td>
              <td className="sub nw">{x.site ?? ""}{x.date ? ` · ${x.date}` : ""}</td>
            </tr>))}
          </tbody>
        </table></div>
      ) : <p className="note">No relevant results.</p>}
      {items.length > 10 && <p className="note">Showing the first 10 of {items.length}; new ones first.</p>}
      <p className="note">Search results from public pages, not an official feed: open the link and check the circular itself before acting.
        New ones also go to Telegram when it is configured.{c.errors?.length ? ` ${c.errors.length} search(es) failed this run.` : ""}</p>
    </Card>
  );
}
