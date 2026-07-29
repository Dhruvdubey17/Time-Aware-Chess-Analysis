"use client";

import type { AnalysisResult } from "@/lib/types";
import { LADDER, labelColor } from "@/lib/labels";

// Per-game overview: the label mix, and an honest sentence about the time-aware
// findings with quick links to the pressure upgrades.
export default function Summary({
  result,
  onJump,
}: {
  result: AnalysisResult;
  onJump: (ply: number) => void;
}) {
  const s = result.summary;
  const counts = s.baseline_counts;
  const ordered = [...LADDER].reverse().filter((l) => counts[l]);

  return (
    <div className="rounded-md border border-border bg-panel p-3">
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
        {ordered.map((l) => (
          <span key={l} className="inline-flex items-center gap-1" style={{ color: labelColor(l) }}>
            <b className="tabular-nums">{counts[l]}</b>
            <span className="text-muted">{l}</span>
          </span>
        ))}
      </div>

      {result.time_aware_available ? (
        <div className="mt-2.5 border-t border-border pt-2.5 text-sm">
          {s.n_upgrades > 0 ? (
            <>
              <p className="text-accent">
                <b>{s.n_upgrades}</b> pressure {s.n_upgrades === 1 ? "upgrade" : "upgrades"}: good
                moves found under real time pressure.
              </p>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {s.upgrades.map((u) => (
                  <button
                    key={u.ply}
                    onClick={() => onJump(u.ply)}
                    className="rounded bg-accent/15 px-1.5 py-0.5 text-xs text-accent hover:bg-accent/25"
                  >
                    {u.move_number}{u.side === "white" ? "." : "..."}{u.san}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <p className="text-muted">
              No pressure upgrades here. Nothing hard enough turned up on a low clock.
            </p>
          )}
        </div>
      ) : (
        <p className="mt-2.5 border-t border-border pt-2.5 text-xs text-muted">
          {result.time_aware_note}
        </p>
      )}
    </div>
  );
}
