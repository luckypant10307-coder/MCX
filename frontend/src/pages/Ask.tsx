import { useEffect, useRef, useState, type FormEvent } from "react";
import { authedFetch } from "../api";
import { useAuth } from "../auth";
import { fdate } from "../format";
import { Card } from "../components/ui";

interface Turn { role: "user" | "assistant"; text: string; tools?: string[]; asof?: string; error?: boolean }

const STARTERS = [
  "Why is each pair in its current status?",
  "What is the margin on GOLDM now, and how high did it go in 2026?",
  "How did the backtest do by year?",
  "Explain the z-score in simple words.",
  "Any new MCX circulars?",
];

// What each tool read, in words a reader recognises.
const READ: Record<string, string> = {
  get_overview: "overview", get_pairs: "pair statuses", get_pair: "pair detail", get_decisions: "decision log",
  get_backtest: "backtest", get_account: "account", get_margins: "margin rates", get_circulars: "circulars",
  get_data_quality: "data quality",
};

/** Questions about the monitor, answered by Gemini from the latest snapshot. */
export default function Ask() {
  const { config } = useAuth();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [left, setLeft] = useState<number | null>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ block: "nearest" }); }, [turns, busy]);

  const ask = async (q: string) => {
    q = q.trim();
    if (!q || busy) return;
    const history = turns.filter((t) => !t.error).map(({ role, text }) => ({ role, text }));
    setTurns((t) => [...t, { role: "user", text: q }]);
    setText("");
    setBusy(true);
    try {
      const r = await authedFetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q, history }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || `The assistant answered with an error (${r.status}).`);
      setTurns((t) => [...t, { role: "assistant", text: j.text, tools: j.tools, asof: j.asof }]);
      if (typeof j.questions_left === "number") setLeft(j.questions_left);
    } catch (e) {
      setTurns((t) => [...t, { role: "assistant", text: (e as Error).message, error: true }]);
    } finally {
      setBusy(false);
    }
  };
  const submit = (e: FormEvent) => { e.preventDefault(); ask(text); };

  if (config && !config.chat_enabled)
    return (
      <>
        <Card title="Ask">
          <p className="note">The assistant is not set up on this server. Set <span className="mono">GEMINI_API_KEY</span> in the
            environment (Render: Environment tab) and redeploy.</p>
        </Card>
      </>);

  return (
    <>
      <Card title="Ask about the monitor" note={left !== null ? `${left} questions left today` : "Gemini · reads the latest snapshot"}>
        <div className="chat" aria-live="polite">
          {turns.length === 0 && (
            <div className="chat-empty">
              <p className="note">Ask why a pair has its status, what margin costs, how the backtest went, or what a term means.
                Answers come from the same data as the dashboard, with the session date.</p>
              <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                {STARTERS.map((s) => <button key={s} className="chipbtn" onClick={() => ask(s)}>{s}</button>)}
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            <div key={i} className={"msg " + t.role + (t.error ? " error" : "")}>
              <div className="bubble">{t.text}</div>
              {t.role === "assistant" && !t.error && (
                <div className="sub">
                  {t.asof ? `As of ${fdate(t.asof)}` : ""}
                  {t.tools?.length ? ` · read: ${[...new Set(t.tools)].map((x) => READ[x] ?? x).join(", ")}` : " · answered without reading data"}
                </div>
              )}
            </div>
          ))}
          {busy && <div className="msg assistant"><div className="bubble typing" aria-label="Thinking">…</div></div>}
          <div ref={end} />
        </div>
        <form className="chat-form" onSubmit={submit}>
          <label htmlFor="ask-input" className="sr-only">Your question</label>
          <input id="ask-input" value={text} onChange={(e) => setText(e.target.value)} maxLength={1000}
            placeholder="e.g. Why is GOLDM/GOLDPETAL quiet?" autoComplete="off" disabled={busy} />
          <button className="btn primary" disabled={busy || !text.trim()}>Ask</button>
        </form>
        <p className="note">The assistant explains what the system shows; it does not give buy or sell advice. Check important numbers
          on the dashboard pages.</p>
      </Card>
    </>
  );
}
