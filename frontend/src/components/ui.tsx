import type { ReactNode } from "react";
import type { UseQueryResult } from "@tanstack/react-query";

export function Kpi({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "warn" | "bad" | "gold" }) {
  return (
    <div className="kpi">
      <div className="l">{label}</div>
      <div className={"v" + (tone ? " c-" + tone : "")}>{value}</div>
      {sub && <div className="s">{sub}</div>}
    </div>
  );
}

export function Card({ title, note, children, className = "" }: { title?: ReactNode; note?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={"card " + className}>
      {title && <h2>{title}{note && <small>{note}</small>}</h2>}
      {children}
    </section>
  );
}

const LABEL: Record<string, string> = { NO_TRADE: "no trade", REJECTED_DATE: "rejected · date", SIGNAL: "passed" };
export const Chip = ({ s, label }: { s: string; label?: string }) => (
  <span className={"chip " + s}>{label ?? LABEL[s] ?? s.replace("_", " ").toLowerCase()}</span>
);

/** Shows a skeleton while loading and a readable error instead of a blank page. */
export function Loaded<T>({ q, children }: { q: UseQueryResult<T>; children: (d: T) => ReactNode }) {
  if (q.isPending) return <div className="skeleton" aria-busy="true" aria-label="Loading" />;
  if (q.isError)
    return (
      <div className="err" role="alert">
        <b>Could not load this page.</b> {(q.error as Error).message}
        <div className="note">If no snapshot exists yet, run scripts/nightly.py on the server.</div>
      </div>
    );
  return <>{children(q.data as T)}</>;
}

export function Legend({ items }: { items: { name: string; color: string; kind?: "line" | "dash" | "box" | "dot" }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.name}>
          <i className={i.kind && i.kind !== "line" ? i.kind : ""}
             style={i.kind === "box" || i.kind === "dot" ? { background: i.color } : { borderColor: i.color }} />
          {i.name}
        </span>
      ))}
    </div>
  );
}

export function TableView({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <details>
      <summary>Table view</summary>
      <div className="tw"><table>
        <thead><tr>{head.map((h, i) => <th key={h} className={i ? "num" : ""}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, j) => <tr key={j}>{r.map((c, i) => <td key={i} className={i ? "num" : ""}>{c}</td>)}</tr>)}</tbody>
      </table></div>
    </details>
  );
}
