import { StrictMode, lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import "./styles.css";

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

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
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
              <Route path="settings" element={<Settings />} />
              <Route path="*" element={<Overview />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
