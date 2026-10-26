import type { Account } from "../api";
import { fdate, fin, lakh, pct } from "../format";

/** Kill-switch state: how far the account is below its peak, against the limit. */
export default function AccountRisk({ a }: { a: Account }) {
  if (!fin(a.drawdown_pct) || !fin(a.limit_pct)) return null;
  const scale = Math.max(a.limit_pct * 2, a.drawdown_pct * 1.15);
  const fill = Math.min(100, (a.drawdown_pct / scale) * 100);
  const mark = (a.limit_pct / scale) * 100;
  return (
    <section className="card">
      <div className="verdict">
        <div className="t">
          <span className={"dot " + (a.halted ? "bad" : "good")} aria-hidden="true">{a.halted ? "■" : "✓"}</span>
          <div>
            <b>{a.halted ? "Kill switch on: new exposure halted" : "Kill switch armed"}</b>
            <div className="sub">
              Account {lakh(a.value_now)}, {pct(a.drawdown_pct, 1)} below its peak of {lakh(a.peak_value)}
              {a.peak_date ? ` on ${fdate(a.peak_date)}` : ""}. Limit {pct(a.limit_pct)} of capital.
              {a.halted ? " Open positions are still managed to their exits; restarting is a decision for a person." : ""}
            </div>
          </div>
        </div>
        <span className={"chip " + (a.halted ? "bad" : "ok")}>{a.halted ? "halted" : "trading allowed"}</span>
      </div>
      <div className="meter" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(a.drawdown_pct * 1000) / 10}
           aria-label={`Drawdown ${pct(a.drawdown_pct, 1)} against a ${pct(a.limit_pct)} limit`}>
        <div className={"fill" + (a.halted ? " bad" : "")} style={{ width: `${fill}%` }} />
        <div className="limit" style={{ left: `${mark}%` }}><span>limit {pct(a.limit_pct)}</span></div>
      </div>
      <p className="note">Drawdown is measured on the walk-forward book from its highest value, starting capital included.
        Worst fall in the history: {lakh(a.max_drawdown_inr)}.</p>
    </section>
  );
}
