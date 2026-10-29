// Sign-in state for the whole app. The server tells the browser at runtime whether
// accounts are switched on (/api/config), so the same build works with or without
// Supabase: without it the dashboard stays open, as a single-user tool.
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js";
import { setTokenGetter } from "./api";

export interface AppConfig {
  auth_enabled: boolean; supabase_url: string | null; supabase_publishable_key: string | null; vapid_public_key: string | null;
  chat_enabled?: boolean;
}
interface AuthState {
  ready: boolean; config: AppConfig | null; supabase: SupabaseClient | null;
  session: Session | null; error: string | null; signOut: () => Promise<void>;
}

const Ctx = createContext<AuthState>({ ready: false, config: null, supabase: null, session: null, error: null, signOut: async () => {} });
export const useAuth = () => useContext(Ctx);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [supabase, setSupabase] = useState<SupabaseClient | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let unsub: (() => void) | undefined;
    fetch("/api/config").then((r) => r.json()).catch(() => ({ auth_enabled: false })).then(async (cfg: AppConfig) => {
      setConfig(cfg);
      try {
        if (!cfg.auth_enabled || !cfg.supabase_url || !cfg.supabase_publishable_key) return;
        const sb = createClient(cfg.supabase_url, cfg.supabase_publishable_key, {
          auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
        });
        setSupabase(sb);
        setTokenGetter(async () => (await sb.auth.getSession()).data.session?.access_token ?? null);
        unsub = sb.auth.onAuthStateChange((_e, s) => setSession(s)).data.subscription.unsubscribe;
        const { data } = await sb.auth.getSession();
        setSession(data.session);
      } catch (e) {
        setError((e as Error).message || String(e));
      } finally {
        setReady(true);            // never leave the app stuck on the loading placeholder
      }
    });
    return () => unsub?.();
  }, []);

  const signOut = async () => { await supabase?.auth.signOut(); setSession(null); };
  return <Ctx.Provider value={{ ready, config, supabase, session, error, signOut }}>{children}</Ctx.Provider>;
}
