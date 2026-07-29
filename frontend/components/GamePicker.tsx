"use client";

import type { GameSummary } from "@/lib/types";

export default function GamePicker({
  games,
  onPick,
  onBack,
}: {
  games: GameSummary[];
  onPick: (index: number) => void;
  onBack: () => void;
}) {
  return (
    <div className="mx-auto max-w-2xl p-6">
      <button onClick={onBack} className="mb-4 text-sm text-muted hover:text-primary">
        ← back
      </button>
      <h1 className="text-xl font-semibold">This file has {games.length} games</h1>
      <p className="mt-1 text-muted">Pick one to review.</p>

      <div className="mt-4 space-y-2">
        {games.map((g) => (
          <button
            key={g.index}
            disabled={!g.accepted}
            onClick={() => onPick(g.index)}
            className={`block w-full rounded-md border border-border p-3 text-left ${
              g.accepted ? "bg-panel hover:border-accent" : "bg-panel/40 opacity-60"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-medium">
                {g.white} vs {g.black}
              </span>
              <span className="text-sm text-muted">{g.result}</span>
            </div>
            <div className="mt-1 text-xs text-faint">
              {g.accepted
                ? `${g.opening || "opening unknown"} · ${g.regime ?? "unknown"} ${g.time_control} · ${g.n_moves} moves`
                : g.reject_reason}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
