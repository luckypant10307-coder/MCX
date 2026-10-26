import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useHealth, useOverview } from "../api";
import { fdate, fday, ist } from "../format";

const I = {
  overview: <path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" />,
  pairs: <path d="M4 8h13l-3-3M20 16H7l3 3" />,
  log: <path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01" />,
  research: <path d="M4 19h16M7 16V9M12 16V5M17 16v-4" />,
  backtest: <path d="M3 17l5-5 4 3 8-8M15 7h5v5" />,
  data: <path d="M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />,
  settings: <path d="M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" />,
};
const NAV: { to: string; label: string; icon: ReactNode; badge?: boolean }[] = [
  { to: "/", label: "Overview", icon: I.overview },
  { to: "/pairs", label: "Pairs", icon: I.pairs, badge: true },
  { to: "/decisions", label: "Decision log", icon: I.log },
  { to: "/research", label: "Research", icon: I.research },
  { to: "/backtest", label: "Backtest", icon: I.backtest },
  { to: "/data", label: "Data & audit", icon: I.data },
  { to: "/settings", label: "Settings", icon: I.settings },
];
const TITLES: Record<string, string> = {
  "/": "Overview", "/pairs": "Pairs", "/decisions": "Decision log", "/research": "Research",
  "/backtest": "Backtest", "/data": "Data & audit", "/settings": "Settings",
};

function useTheme(): [boolean, () => void] {
  const sysDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
  const [dark, setDark] = useState<boolean>(() => {
    try { const t = localStorage.getItem("theme"); if (t) return t === "dark"; } catch { /* storage blocked */ }
    return sysDark();
  });
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch { /* storage blocked */ }
  }, [dark]);
  return [dark, () => setDark((d) => !d)];
}

export default function Layout() {
  const { pathname } = useLocation();
  const health = useHealth();
  const ov = useOverview();
  const [dark, toggle] = useTheme();
  const h = health.data;
  const alerts = ov.data?.summary.ALERT ?? 0;
  const title = pathname.startsWith("/pairs/") ? "Pair detail" : TITLES[pathname] ?? "Gold Spread Monitor";
  useEffect(() => { document.title = `${title} · Gold Spread Monitor`; window.scrollTo(0, 0); }, [title]);

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <div className="logo">Au</div>
          <div><b>Gold Spread Monitor</b><small>Binary Bridge · prototype</small></div>
        </div>
        <nav className="nav" aria-label="Sections">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}
              className={({ isActive }) => (isActive || (n.to === "/pairs" && pathname.startsWith("/pairs")) ? "active" : "")}>
              <svg viewBox="0 0 24 24" aria-hidden="true">{n.icon}</svg>{n.label}
              {n.badge && ov.data && <span className={"badge" + (alerts ? "" : " calm")}>{alerts}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="rail-foot">
          {h ? (<>
            Data: {h.sample_data ? "synthetic sample in MCX bhavcopy format" : "MCX bhavcopy via mtf.trading"}<br />
            Engine: rule-based, no AI in the decision<br />Build {fdate(h.generated_at)}
          </>) : health.isError ? "API unreachable" : "Connecting…"}
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <h1>{title}</h1>
          {h && <span className={"pill " + (h.sample_data ? "sample" : "real")}>{h.sample_data ? "sample data" : "MCX data"}</span>}
          <div className="right">
            {h && <span>Last run {ist(h.generated_at)} IST · run <span className="mono">{h.run_id}</span></span>}
            {h && <span className="datechip">{fday(h.asof)}</span>}
            <button className={"toggle" + (dark ? " on" : "")} onClick={toggle} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
              aria-pressed={dark}><i aria-hidden="true">{dark ? "☾" : "☀"}</i></button>
          </div>
        </header>
        <Outlet />
      </main>
    </div>
  );
}
