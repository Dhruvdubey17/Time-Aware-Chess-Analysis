"use client";

import { useState } from "react";
import { chesscomGames } from "@/lib/api";
import type { ChessComGamesResponse } from "@/lib/types";

// Shows a fetched month of chess.com games so the user can pick one. The month
// dropdown re-fetches other months from the same account.
export default function ChessComPicker({
  initial,
  username,
  locked = false,
  onPick,
  onBack,
}: {
  initial: ChessComGamesResponse;
  username: string;
  locked?: boolean;
  onPick: (pgn: string) => void;
  onBack: () => void;
}) {
  const [resp, setResp] = useState(initial);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const changeMonth = async (month: string) => {
    setError(null);
    setLoading(true);
    try {
      setResp(await chesscomGames(username, month));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load that month.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl p-6">
      {/* When locked to a launch username, the games page is home, so there is
          nowhere to go back to. */}
      {!locked && (
        <button onClick={onBack} className="mb-4 text-sm text-muted hover:text-primary">
          ← back
        </button>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold">{username}&apos;s games</h1>
        <label className="text-sm text-muted">
          Month{" "}
          <select
            value={resp.month}
            onChange={(e) => changeMonth(e.target.value)}
            className="rounded-md border border-border bg-panel p-1.5 text-primary focus:border-accent focus:outline-none"
          >
            {resp.months.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && <p className="mt-3 text-sm text-bad">{error}</p>}

      <div className="mt-4 space-y-2">
        {loading ? (
          <p className="text-muted">Loading…</p>
        ) : (
          resp.games.map((g, i) => (
            <button
              key={g.url || i}
              onClick={() => onPick(g.pgn)}
              className="block w-full rounded-md border border-border bg-panel p-3 text-left hover:border-accent"
            >
              <div className="flex items-center justify-between">
                <span className="font-medium">
                  {g.white} <span className="text-faint">({g.white_elo ?? "?"})</span> vs{" "}
                  {g.black} <span className="text-faint">({g.black_elo ?? "?"})</span>
                </span>
                <span className="text-sm text-muted">{g.result}</span>
              </div>
              <div className="mt-1 text-xs text-faint">
                {g.time_class} {g.time_control} · {g.date}
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
