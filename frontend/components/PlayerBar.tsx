"use client";

import { clockStr } from "@/lib/chess";

// One player's identity line, shown above and below the board. Minimal: a small
// color swatch, name, rating, and the clock as a pill on the right.
export default function PlayerBar({
  name,
  rating,
  clock,
  color,
}: {
  name: string;
  rating: number | null;
  clock: number | null;
  color: "white" | "black";
}) {
  return (
    <div className="flex items-center gap-2 py-1">
      <div
        className="h-6 w-6 shrink-0 rounded border border-border"
        style={{ background: color === "white" ? "#e8e6df" : "#39373400" }}
      >
        <div
          className="h-full w-full rounded"
          style={{ background: color === "white" ? "#e8e6df" : "#2b2926" }}
        />
      </div>
      <span className="truncate font-medium">{name || "?"}</span>
      {rating != null && <span className="text-sm text-faint">({rating})</span>}
      {clock != null && (
        <span className="ml-auto rounded bg-panel2 px-2 py-0.5 text-sm font-semibold tabular-nums text-primary">
          {clockStr(clock)}
        </span>
      )}
    </div>
  );
}
