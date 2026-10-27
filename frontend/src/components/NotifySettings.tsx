import { useEffect, useState } from "react";
import { useAuth } from "../auth";
import { authedFetch } from "../api";
import { currentSubscription, disablePush, enablePush, needsInstallForPush, pushSupported } from "../push";
import { Card } from "./ui";

interface Prefs { trade_alerts: boolean; digest: boolean; circulars: boolean; pairs: string[] | null }
const PAIRS = ["GOLDM/GOLDTEN", "GOLDM/GOLDGUINEA", "GOLDM/GOLDPETAL", "GOLDTEN/GOLDGUINEA", "GOLDTEN/GOLDPETAL", "GOLDGUINEA/GOLDPETAL"];
const TOGGLES: [keyof Prefs, string, string][] = [
  ["trade_alerts", "Trade alerts", "A pair becomes tradeable, or an alert clears. Loud on your phone."],
  ["digest", "Evening summary", "Unusual points that were not traded, and data to review. Silent."],
  ["circulars", "MCX circulars", "New margin or contract circulars found on the web."],
];

function Switch({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return <button className={"toggle switch" + (on ? " on" : "")} role="switch" aria-checked={on} aria-label={label} onClick={() => onChange(!on)}><i aria-hidden="true" /></button>;
}

/** One place to turn notifications on for this device and choose what to hear about. */
export default function NotifySettings() {
  const { supabase, session, config, signOut } = useAuth();
  const [prefs, setPrefs] = useState<Prefs | null>(null);
  const [device, setDevice] = useState<"on" | "off" | "unsupported">("off");
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const uid = session?.user.id;

  useEffect(() => {
    if (!supabase || !uid) return;
    supabase.from("notification_prefs").select("*").eq("user_id", uid).maybeSingle().then(({ data }) => setPrefs(data as Prefs));
    if (!pushSupported()) setDevice("unsupported");
    else currentSubscription().then((s) => setDevice(s ? "on" : "off"));
  }, [supabase, uid]);

  if (!supabase || !uid) return null;

  const save = async (patch: Partial<Prefs>) => {
    const next = { ...(prefs as Prefs), ...patch };
    setPrefs(next);
    const { error } = await supabase.from("notification_prefs").update({ ...patch, updated_at: new Date().toISOString() }).eq("user_id", uid);
    setStatus(error ? "Could not save: " + error.message : "Saved.");
  };
  const toggleDevice = async () => {
    setBusy(true); setStatus(null);
    try {
      if (device === "on") { await disablePush(supabase); setDevice("off"); setStatus("Notifications turned off on this device."); }
      else { await enablePush(supabase, config!.vapid_public_key!, uid); setDevice("on"); setStatus("Notifications are on for this device."); }
    } catch (e) { setStatus((e as Error).message); }
    setBusy(false);
  };
  const test = async () => {
    setBusy(true);
    const r = await authedFetch("/api/notifications/test", { method: "POST" });
    const j = await r.json().catch(() => ({}));
    setStatus(r.ok ? (j.pushed ? `Test sent to ${j.pushed} device(s). It is also in the bell.` : "Test added to the bell. No device has notifications on yet.")
                   : j.detail || "Could not send a test.");
    setBusy(false);
  };
  const followAll = !prefs?.pairs?.length;

  return (
    <Card title="Notifications" note={session?.user.email ?? ""}>
      <div className="notify-hero">
        <div>
          <b>{device === "on" ? "On for this device" : device === "unsupported" ? "This browser cannot receive notifications" : "Get alerts on this device"}</b>
          <div className="sub">
            {device === "unsupported" ? "Try Chrome, Edge or Firefox, or install the app on your phone."
              : needsInstallForPush() ? "On iPhone: tap Share, then Add to Home Screen, open the app from there, then turn this on."
              : "Pop-up alerts on this phone or computer, even when the site is closed."}
          </div>
        </div>
        {device !== "unsupported" && !needsInstallForPush() &&
          <button className={"btn" + (device === "on" ? "" : " primary")} disabled={busy || !config?.vapid_public_key} onClick={toggleDevice}>
            {device === "on" ? "Turn off" : "🔔 Turn on notifications"}</button>}
      </div>

      <div className="pref-list">{TOGGLES.map(([k, name, what]) => (
        <div className="pref" key={k}><div><b>{name}</b><div className="sub">{what}</div></div>
          <Switch on={!!prefs?.[k]} label={name} onChange={(v) => save({ [k]: v } as Partial<Prefs>)} /></div>))}
      </div>

      <div><b>Pairs to follow</b><div className="sub">Trade alerts only for these pairs.</div>
        <div className="row" style={{ marginTop: 10, gap: 10 }}>
          <button className={"chipbtn" + (followAll ? " on" : "")} onClick={() => save({ pairs: null })}>All pairs</button>
          {PAIRS.map((p) => {
            const on = !followAll && prefs!.pairs!.includes(p);
            return <button key={p} className={"chipbtn" + (on ? " on" : "")} onClick={() => {
              const cur = followAll ? [] : [...prefs!.pairs!];
              const next = on ? cur.filter((x) => x !== p) : [...cur, p];
              save({ pairs: next.length ? next : null });
            }}>{p.replace("/", " / ")}</button>;
          })}
        </div>
      </div>

      <div className="row">
        <button className="btn" onClick={test} disabled={busy}>Send me a test</button>
        <button className="btn" onClick={signOut}>Sign out</button>
        {status && <span className="sub" role="status">{status}</span>}
      </div>
    </Card>
  );
}
