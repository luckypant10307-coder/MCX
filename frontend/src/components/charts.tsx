// Thin wrappers over Recharts so every chart shares one look: hairline grid, muted
// axes, 2px lines, and a soft raised tooltip that lists every series at the hovered x.
import type { ReactNode } from "react";
import { ResponsiveContainer, Tooltip } from "recharts";
import type { Palette } from "../theme";

export const axis = (p: Palette) => ({
  stroke: p.grid, tick: { fill: p.muted, fontSize: 11.5 }, tickLine: false, axisLine: { stroke: p.grid },
});

export function Box({ height, children, label }: { height: number; children: ReactNode; label: string }) {
  return (
    <div className="chartbox" role="img" aria-label={label}>
      <ResponsiveContainer width="100%" height={height}>{children as React.ReactElement}</ResponsiveContainer>
    </div>
  );
}

type Row = { label: string; value: string; color?: string; dash?: boolean };
// What Recharts hands a custom tooltip, narrowed to the fields used here.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TipPayload = { value?: unknown; dataKey?: unknown; color?: string; payload?: any }[] | undefined;
export function SoftTooltip({ head, rows }: { head: (label: unknown) => string; rows: (payload: TipPayload) => Row[] }) {
  return (
    <Tooltip
      cursor={{ strokeOpacity: 0.35 }}
      content={({ active, label, payload }) =>
        active && payload && payload.length ? (
          <div className="tooltip">
            <div className="h">{head(label)}</div>
            {rows(payload).map((r) => (
              <div className="r" key={r.label}>
                {r.color && <i style={{ borderColor: r.color, borderTopStyle: r.dash ? "dashed" : "solid" }} />}
                <b>{r.value}</b><span className="sub">{r.label}</span>
              </div>
            ))}
          </div>
        ) : null
      }
    />
  );
}
