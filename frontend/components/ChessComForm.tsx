"use client";

import { useState } from "react";
import { chesscomGame, chesscomGames } from "@/lib/api";
import type { ChessComGame, ChessComGamesResponse } from "@/lib/types";

// The chess.com fetch panel. This is the app's only network call. It loads an
// account's games (chess.com's API keeps the move clocks that its manual PGN
// export drops), or resolves one game by link.
export default function ChessComForm({
  onGames,
  onGame,
}: {
  onGames: (resp: ChessComGamesResponse, username: string) => void;
  onGame: (game: ChessComGame) => void;
}) {
  const [username, setUsername] = useState("");
  const [link, setLink] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setError(null);
    const user = username.trim();
    if (!user) {
      setError("Enter a chess.com username.");
      return;
    }
    setLoading(true);
    try {
      if (link.trim()) {
        onGame((await chesscomGame(user, link.trim())).game);
      } else {
        onGames(await chesscomGames(user), user);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not fetch from chess.com.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <p className="rounded-md border border-border bg-panel/60 p-3 text-sm text-muted">
        Optional. This is the only time the app uses the network, and only to fetch
        the game you ask for. Pasting and uploading stay fully offline.
      </p>

      <label className="mt-4 block text-sm text-muted">
        chess.com username
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="e.g. magnuscarlsen"
          className="mt-1 w-full rounded-md border border-border bg-panel p-2.5 text-primary placeholder:text-faint focus:border-accent focus:outline-none"
        />
      </label>

      <label className="mt-3 block text-sm text-muted">
        Game link (optional)
        <input
          value={link}
          onChange={(e) => setLink(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="https://www.chess.com/game/live/..."
          className="mt-1 w-full rounded-md border border-border bg-panel p-2.5 text-primary placeholder:text-faint focus:border-accent focus:outline-none"
        />
      </label>

      {error && <p className="mt-2 text-sm text-bad">{error}</p>}

      <button
        onClick={run}
        disabled={loading}
        className="mt-4 rounded-md bg-accent px-5 py-2 font-medium text-app hover:brightness-110 disabled:opacity-40"
      >
        {loading ? "Fetching…" : link.trim() ? "Fetch game" : "Load games"}
      </button>
    </div>
  );
}
