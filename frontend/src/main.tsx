import { StrictMode, lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Login from "./pages/Login";
import { AuthProvider, useAuth } from "./auth";
import "./styles.css";

/** With accounts on, nothing but the login page is shown until someone signs in. */
function Gate({ children }: { children: ReactNode }) {
  const { ready, config, session, error } = useAuth();
  if (!ready) return <div className="login"><div className="skeleton" style={{ width: 360 }} aria-busy="true" /></div>;
  if (error) return <div className="login"><div className="err" role="alert"><b>Sign-in could not start.</b> {error}</div></div>;
  if (config?.auth_enabled && !session) return <Login />;
  return <>{children}</>;
}

// Installable app + push: the service worker lives at the site root.
if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => { navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => {}); });
}

// The snapshot changes once a day, so data stays fresh for minutes and every page
// switch is instant after the first load.
const client = new QueryClient({ defaultOptions: { queries: { staleTime: 5 * 60_000, retry: 1, refetchOnWindowFocus: false } } });

// Pages beyond the first load on demand, keeping the initial download small.
const Pairs = lazy(() => import("./pages/Pairs"));
const PairDetail = lazy(() => import("./pages/PairDetail"));
const Decisions = lazy(() => import("./pages/Decisions"));
const Research = lazy(() => import("./pages/Research"));
const Backtest = lazy(() => import("./pages/Backtest"));
const DataAudit = lazy(() => import("./pages/DataAudit"));
const Settings = lazy(() => import("./pages/Settings"));
const Ask = lazy(() => import("./pages/Ask"));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <AuthProvider><Gate>
      <BrowserRouter>
        <Suspense fallback={<div className="skeleton" aria-busy="true" />}>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Overview />} />
              <Route path="pairs" element={<Pairs />} />
              <Route path="pairs/:id" element={<PairDetail />} />
              <Route path="decisions" element={<Decisions />} />
              <Route path="research" element={<Research />} />
              <Route path="backtest" element={<Backtest />} />
              <Route path="data" element={<DataAudit />} />
              <Route path="ask" element={<Ask />} />
              <Route path="settings" element={<Settings />} />
              <Route path="*" element={<Overview />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
      </Gate></AuthProvider>
    </QueryClientProvider>
  </StrictMode>,
);
