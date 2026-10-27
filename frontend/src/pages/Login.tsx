import { useState, type FormEvent } from "react";
import { useAuth } from "../auth";

// Supabase phrases the allow-list refusal as a database error; show it as what it is.
const friendly = (msg: string) =>
  /invitation only|database error saving new user/i.test(msg) ? "This email is not on the invite list. Ask the admin to add it."
    : /invalid login credentials/i.test(msg) ? "Email or password is wrong."
    : /email not confirmed/i.test(msg) ? "Confirm your email first: open the link we sent you."
    : msg;

export default function Login() {
  const { supabase } = useAuth();
  const [mode, setMode] = useState<"in" | "up">("in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "err" | "ok"; text: string } | null>(null);

  const google = async () => {
    setMsg(null);
    const { error } = await supabase!.auth.signInWithOAuth({ provider: "google", options: { redirectTo: window.location.origin } });
    if (error) setMsg({ kind: "err", text: friendly(error.message) });
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true); setMsg(null);
    const res = mode === "in"
      ? await supabase!.auth.signInWithPassword({ email, password })
      : await supabase!.auth.signUp({ email, password, options: { emailRedirectTo: window.location.origin } });
    setBusy(false);
    if (res.error) return setMsg({ kind: "err", text: friendly(res.error.message) });
    if (mode === "up" && !res.data.session) setMsg({ kind: "ok", text: "Account created. Check your inbox for a confirmation link, then sign in." });
  };

  return (
    <div className="login">
      <div className="card login-card">
        <div className="brand" style={{ justifyContent: "center" }}>
          <div className="logo">Au</div>
          <div><b>Gold Spread Monitor</b><small>Binary Bridge · prototype</small></div>
        </div>
        <h2 style={{ justifyContent: "center", fontSize: 21 }}>{mode === "in" ? "Sign in" : "Create your account"}</h2>
        <button className="btn wide" onClick={google} type="button">
          <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true"><path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.6 5.4 2.7 13.3l7.9 6.1C12.5 13.6 17.8 9.5 24 9.5z"/><path fill="#4285F4" d="M46.1 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h12.4c-.5 2.9-2.2 5.3-4.6 6.9l7.4 5.8c4.3-4 6.9-9.9 6.9-16.7z"/><path fill="#FBBC05" d="M10.6 28.6c-.5-1.4-.8-3-.8-4.6s.3-3.1.8-4.6l-7.9-6.1C1 16.6 0 20.2 0 24s1 7.4 2.7 10.7l7.9-6.1z"/><path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.4-5.8c-2.1 1.4-4.8 2.2-8.5 2.2-6.2 0-11.5-4.1-13.4-9.8l-7.9 6.1C6.6 42.6 14.6 48 24 48z"/></svg>
          Continue with Google
        </button>
        <div className="divider"><span>or with email</span></div>
        <form className="login-form" onSubmit={submit}>
          <label>Email<input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} id="login-email" /></label>
          <label>Password<input type="password" autoComplete={mode === "in" ? "current-password" : "new-password"} required minLength={8}
            value={password} onChange={(e) => setPassword(e.target.value)} id="login-password" /></label>
          <button className="btn wide primary" disabled={busy} type="submit">{busy ? "Please wait…" : mode === "in" ? "Sign in" : "Create account"}</button>
        </form>
        {msg && <div className={"login-msg " + msg.kind} role="status">{msg.text}</div>}
        <p className="note" style={{ textAlign: "center" }}>
          {mode === "in" ? "New here? " : "Already registered? "}
          <button className="linkbtn" type="button" onClick={() => { setMode(mode === "in" ? "up" : "in"); setMsg(null); }}>
            {mode === "in" ? "Create an account" : "Sign in"}</button>
        </p>
        <p className="note" style={{ textAlign: "center" }}>Registration is by invitation. Research signals, not investment advice.</p>
      </div>
    </div>
  );
}
