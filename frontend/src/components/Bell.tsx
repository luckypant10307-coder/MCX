import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth";

interface Note { id: number; kind: string; title: string; body: string; url: string | null; created_at: string; read_at: string | null }

const ICON: Record<string, string> = { new_alert: "🚨", alert_cleared: "✅", digest: "📝", circulars: "📄", no_data: "⚠️", data_changed_same_session: "⚠️", test: "🔔" };
const ago = (iso: string) => {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago` : s < 86400 ? `${Math.floor(s / 3600)} h ago` : new Date(iso).toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
};

/** The notification centre: every message the user was sent, read or not. Live via Supabase Realtime. */
export default function Bell() {
  const { supabase, session } = useAuth();
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<Note[]>([]);
  const box = useRef<HTMLDivElement>(null);
  const go = useNavigate();
  const uid = session?.user.id;

  const load = useCallback(async () => {
    if (!supabase || !uid) return;
    const { data } = await supabase.from("notifications").select("*").order("created_at", { ascending: false }).limit(30);
    setItems((data as Note[]) ?? []);
  }, [supabase, uid]);

  useEffect(() => {
    load();
    if (!supabase || !uid) return;
    const ch = supabase.channel("notes-" + uid)
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "notifications", filter: `user_id=eq.${uid}` },
        (p) => setItems((cur) => [p.new as Note, ...cur].slice(0, 30)))
      .subscribe();
    return () => { supabase.removeChannel(ch); };
  }, [supabase, uid, load]);

  useEffect(() => {
    const close = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", close); document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, []);

  if (!supabase || !uid) return null;
  const unread = items.filter((n) => !n.read_at).length;
  const markRead = async (ids: number[]) => {
    if (!ids.length) return;
    const now = new Date().toISOString();
    setItems((cur) => cur.map((n) => (ids.includes(n.id) ? { ...n, read_at: now } : n)));
    await supabase.from("notifications").update({ read_at: now }).in("id", ids);
  };

  return (
    <div className="bell" ref={box}>
      <button className="bell-btn" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-label={`Notifications, ${unread} unread`}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.9 1.9 0 0 0 3.4 0" /></svg>
        {unread > 0 && <span className="bell-count">{unread > 9 ? "9+" : unread}</span>}
      </button>
      {open && (
        <div className="bell-panel" role="dialog" aria-label="Notifications">
          <div className="bell-head"><b>Notifications</b>
            {unread > 0 && <button className="linkbtn" onClick={() => markRead(items.filter((n) => !n.read_at).map((n) => n.id))}>Mark all read</button>}</div>
          {items.length ? (
            <ul>{items.map((n) => (
              <li key={n.id} className={n.read_at ? "" : "unread"}>
                <button onClick={() => { markRead([n.id]); setOpen(false); if (n.url) go(n.url); }}>
                  <span className="bell-ic" aria-hidden="true">{ICON[n.kind] ?? "🔔"}</span>
                  <span className="bell-txt"><b>{n.title}</b><span className="clamp2">{n.body}</span><small>{ago(n.created_at)}</small></span>
                </button>
              </li>))}
            </ul>
          ) : <p className="note" style={{ padding: "8px 4px" }}>Nothing yet. Trade alerts, evening summaries and new circulars will appear here.</p>}
          <button className="btn wide" onClick={() => { setOpen(false); go("/settings"); }}>Notification settings</button>
        </div>
      )}
    </div>
  );
}
