// Recharts writes colours as SVG attributes, where CSS variables are not reliable, so
// charts read the resolved token values here and re-render when the theme flips.
import { useEffect, useState } from "react";

const TOKENS = ["ink", "ink-2", "muted", "grid", "gold", "gold-soft", "s1", "s2", "s3",
  "o1", "o2", "o3", "o4", "pos", "neg", "good", "warn", "band", "surface"] as const;
export type Palette = Record<(typeof TOKENS)[number], string>;

function read(): Palette {
  const cs = getComputedStyle(document.documentElement);
  return Object.fromEntries(TOKENS.map((t) => [t, cs.getPropertyValue("--" + t).trim()])) as Palette;
}

export function usePalette(): Palette {
  const [p, setP] = useState<Palette>(read);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setP(read());
    mq.addEventListener("change", update);
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => { mq.removeEventListener("change", update); obs.disconnect(); };
  }, []);
  return p;
}
