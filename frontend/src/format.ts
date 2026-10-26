// Formatting in Indian conventions: ₹ with lakh grouping, bp for spreads, IST times.
export const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const sign = (v: number) => (v > 0 ? "+" : v < 0 ? "−" : "");

export const bp = (v: unknown, d = 1) => (fin(v) ? `${sign(v)}${Math.abs(v).toFixed(d)} bp` : "–");
export const num = (v: unknown, d = 2) => (fin(v) ? `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}` : "–");
export const int = (v: unknown) => (fin(v) ? Math.round(v).toLocaleString("en-IN") : "–");
export const inr = (v: unknown) => (fin(v) ? `${v < 0 ? "−" : ""}₹${Math.abs(Math.round(v)).toLocaleString("en-IN")}` : "–");
export const lakh = (v: unknown) => (fin(v) ? `${v < 0 ? "−" : ""}₹${(Math.abs(v) / 1e5).toFixed(2)} L` : "–");
export const pct = (v: unknown, d = 0) => (fin(v) ? `${(v * 100).toFixed(d)}%` : "–");
/** 1,90,495 -> 1.9 L; 34,736 -> 34.7 K */
export const compact = (v: unknown) =>
  !fin(v) ? "–" : v >= 1e5 ? `${(v / 1e5).toFixed(1)} L` : v >= 1e3 ? `${(v / 1e3).toFixed(1)} K` : int(v);

const d0 = (s: string) => new Date(s.slice(0, 10) + "T00:00:00");
export const fdate = (s?: string | null) =>
  s ? d0(s).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }) : "–";
export const fexp = (s?: string | null) =>
  s ? d0(s).toLocaleDateString("en-GB", { day: "2-digit", month: "short" }).toUpperCase() : "–";
export const fday = (s?: string | null) =>
  s ? d0(s).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric" }) : "–";
export const ist = (s?: string | null, withDate = false) =>
  s ? new Date(s).toLocaleString("en-IN", {
        timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false,
        ...(withDate ? { day: "2-digit", month: "short" } : {}),
      }) : "–";
export const ms = (s: string) => d0(s).getTime();
export const fmon = (t: number) => new Date(t).toLocaleDateString("en-GB", { month: "short", year: "2-digit" });
export const vs = (pair: string) => pair.replace("/", " vs ");
export const pairUrl = (pair: string) => pair.replace("/", "-");
