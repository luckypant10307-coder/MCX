// Typed access to the FastAPI snapshot API. Every number the UI shows comes from here;
// nothing is computed from market data in the browser.
import { useQuery } from "@tanstack/react-query";

const BASE = (new URLSearchParams(location.search).get("api") || import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

// Set by the auth provider once a user signs in; every API call then carries the token.
let tokenGetter: () => Promise<string | null> = async () => null;
export const setTokenGetter = (fn: () => Promise<string | null>) => { tokenGetter = fn; };

export async function authedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const token = await tokenGetter();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", "Bearer " + token);
  return fetch(BASE + path, { ...init, headers });
}

async function get<T>(path: string): Promise<T> {
  const r = await authedFetch(path);
  if (!r.ok) {
    const detail = await r.json().then((j) => j.detail).catch(() => null);
    throw new Error(detail || `${r.status} ${r.statusText}`);
  }
  return r.json() as Promise<T>;
}

export type Status = "ALERT" | "NO_TRADE" | "QUIET" | "WARMUP";

export interface Meta {
  asof: string; generated_at: string; source: string; run_id: string; sample_data: boolean;
  start: string; end: string; sessions: number; z_entry: number; span_parameters: string;
  data_fingerprint: string; decision_window_sessions?: number;
}
export interface Health { ok: boolean; asof: string; generated_at: string; run_id: string; source: string; sample_data: boolean }
export interface Contract {
  symbol: string; contract_id: string; expiry_date: string; close: number; quote_base_g: number;
  purity: number; norm_close_10g: number; inr_per_g_fine: number; volume: number;
  sessions_to_expiry: number; avg_volume_20: number | null; liquidity: "GREEN" | "YELLOW" | "RED" | null;
}
export interface Leg {
  symbol: string; contract_id?: string; expiry_date?: string; sessions_to_expiry?: number;
  liquidity?: string | null; avg_volume_20?: number | null; tradable?: boolean | null; front_month?: boolean;
}
export interface Risk {
  action?: string; reason?: string; lots_a?: number; lots_b?: number; marginal_margin?: number;
  utilisation_if_added?: number; buffer_state?: string; span_change_pct?: number; floor_binds?: boolean;
  notional_exposure?: number; residual_frac?: number;
}
export interface TradeLeg { side: "BUY" | "SELL"; symbol: string; expiry?: string; lots?: number; fine_g?: number }
export interface Pair {
  pair: string; sym_a: string; sym_b: string; maturity_matched: boolean; spread_bp: number | null;
  centre_bp: number | null; scale_bp: number | null; z: number | null; dislocation_bp: number | null;
  halflife: number | null; adf_t: number | null; gates: Record<string, boolean | null>; signal_ok: boolean;
  direction: number | null; status: Status; reason: string; risk?: Risk | null; legs?: Leg[];
  expiry_gap_days?: number | null; trade?: { legs: TradeLeg[]; margin?: number; residual_frac?: number } | null;
  telegram?: string | null;
}
export interface HistoryRow { date: string; spread_bp: number | null; centre: number | null; scale: number | null; z: number | null; signal_ok: boolean }
export interface PairDetail extends Pair { history: HistoryRow[]; z_entry: number }
export interface Account {
  capital?: number; value_now?: number; peak_value?: number; peak_date?: string | null;
  drawdown_inr?: number; drawdown_pct?: number; limit_pct?: number; halted?: boolean; max_drawdown_inr?: number;
}
export interface Overview {
  meta: Meta; summary: Record<Status, number> & { pairs: number }; contracts: Contract[];
  verdict: { headline: string; edge_survives_costs: boolean | null };
  gold_level: { date: string; inr_per_10g_fine: number }[];
  account: Account;
}
export interface AnomalyRow {
  date: string; symbol: string; contract_id: string; score: number; top_feature: string; top_z: number;
  log_ret: number | null; peer_dev_bp: number | null; ret_in_ticks: number | null; d_log_volume: number | null;
}
export interface Anomalies {
  available: boolean; reason?: string; rows_scored?: number; flagged?: number; flagged_new?: number; flagged_known?: number;
  by_symbol?: { symbol: string; rows: number; flagged: number; flag_rate: number; flagged_already_known: number; flagged_new: number }[];
  by_feature?: { top_feature: string; n: number }[]; top?: AnomalyRow[];
}
export interface Decision { date: string; pair: string; z: number; spread_bp: number; dislocation_bp: number; status: string; reason: string }
export interface Decisions { total: number; counts: Record<string, number>; items: Decision[] }
export interface EsRow { z_band: string; horizon: number; n: number; mean_conv_bp: number; hit_rate: number; t_newey_west: number; boot_ci_lo: number; boot_ci_hi: number; net_conv_bp: number; ci_clears_costs: boolean }
export interface YearRow { year: number; n: number; gross_bp: number; net_bp: number; hit_rate_net: number; net_pnl: number }
export interface Performance {
  n_trades?: number; mean_gross_bp?: number; mean_net_bp?: number; hit_rate_net?: number; gross_pnl?: number;
  total_cost?: number; net_pnl?: number; mean_sessions_held?: number; sharpe?: number; peak_margin?: number;
  max_drawdown_inr?: number; tstat_caveat?: string;
}
export interface Coverage { symbol: string; first: string; last: string; sessions: number; contracts: number; untradeable_rows: number }
export interface Research {
  window: { start: string; end: string; sessions: number; years: number };
  coverage: Coverage[];
  event_study?: { round_trip_bp: number; half_spread_bp: number; rows: EsRow[] };
  costs: { sweep: { half_spread_bp_per_leg: number; statutory_bp: number; execution_bp: number; round_trip_bp: number }[]; gross_bp: number; net_bp: number; breakeven_half_spread_bp: number };
  premiums?: { date: string; symbol: string; premium_bp: number }[];
  by_year?: YearRow[]; by_pair?: (YearRow & { pair: string })[]; exit_reasons?: Record<string, number>;
  equity?: { date: string; equity: number }[];
  verdict: { headline: string; edge_survives_costs: boolean | null }; performance: Performance; meta: Meta;
}
export interface Trade { pair: string; entry_date: string; exit_date: string; lots_a: number; lots_b: number; entry_z: number; gross_bp: number; net_bp: number; exit_reason: string }
export interface Backtest { performance: Performance; trades: Trade[]; verdict: { headline: string } }
export interface ArchiveRow { file: string; requested: string | null; returned: string | null; gold_rows: number; status: string; sha256?: string; saved_at?: string }
export interface DataQuality {
  rows_in: number; rows_out: number; flags: Record<string, number>; run_id: string; data_fingerprint: string;
  asof: string; archive?: { files: number; rejected: number; rows: ArchiveRow[] } | null; coverage: Coverage[];
}
export interface Parameter { group: string; name: string; value: number; unit: string; what: string }
export interface Settings { parameters: Parameter[]; z_entry: number; pair_history: Record<string, { date: string; z: number | null }[]> }
export interface RiskRow extends Risk { pair: string }

const q = <T,>(key: string) => () => useQuery({ queryKey: [key], queryFn: () => get<T>(key) });
export const useHealth = q<Health>("/api/health");
export const useOverview = q<Overview>("/api/overview");
export const usePairs = q<Pair[]>("/api/pairs");
export const useResearch = q<Research>("/api/research");
export const useBacktest = q<Backtest>("/api/backtest");
export const useDataQuality = q<DataQuality>("/api/data-quality");
export const useSettings = q<Settings>("/api/settings");
export const useRisk = q<RiskRow[]>("/api/risk");
export const useAnomalies = q<Anomalies>("/api/anomalies");

export interface RiskFileDecision { at: string; file: string; status: "ACCEPTED" | "REJECTED"; business_date?: string; kind?: string; reason?: string; commodities?: string[] }
export interface RiskFiles {
  accepted: number; rejected: number; waiting: number; recent: RiskFileDecision[]; dates: string[];
  coverage?: { sessions: number; covered: number; coverage: number; first_file: string; last_file: string; gaps: number };
}
export interface Circular { title: string | null; url: string; site?: string | null; snippet?: string | null; date?: string | null; topics: string[]; new: boolean }
export interface Circulars { available: boolean; reason?: string; checked_at?: string; items?: Circular[]; new?: number; errors?: string[] }
export const useRiskFiles = q<RiskFiles>("/api/risk-files");
export const useCirculars = q<Circulars>("/api/circulars");
export const useDecisions = () => useQuery({ queryKey: ["decisions"], queryFn: () => get<Decisions>("/api/decisions?limit=300") });
export const usePair = (id: string) => useQuery({ queryKey: ["pair", id], queryFn: () => get<PairDetail>("/api/pairs/" + encodeURIComponent(id)) });
