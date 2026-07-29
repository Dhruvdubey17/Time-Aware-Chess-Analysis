"use client";

import { useState } from "react";
import type { ChessComGame, ChessComGamesResponse } from "@/lib/types";
import ChessComForm from "./ChessComForm";
import PgnInput from "./PgnInput";

type Source = "paste" | "chesscom";

export default function Landing({
  onPasteSubmit,
  onGames,
  onGame,
  pasteError,
}: {
  onPasteSubmit: (pgn: string) => void;
  onGames: (resp: ChessComGamesResponse, username: string) => void;
  onGame: (game: ChessComGame) => void;
  pasteError?: string | null;
}) {
  const [source, setSource] = useState<Source>("paste");

  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col justify-center p-6">
      <h1 className="text-2xl font-semibold">Chess Review</h1>
      <p className="mt-1 text-muted">
        Get a normal review plus a time-aware second opinion that rewards good moves
        found under real time pressure. Everything runs on your machine.
      </p>

      <div className="mt-5 flex gap-1 rounded-md border border-border bg-panel p-1 text-sm">
        <Tab active={source === "paste"} onClick={() => setSource("paste")}>
          Paste or upload
        </Tab>
        <Tab active={source === "chesscom"} onClick={() => setSource("chesscom")}>
          From chess.com
        </Tab>
      </div>

      <div className="mt-4">
        {source === "paste" ? (
          <PgnInput onSubmit={onPasteSubmit} error={pasteError} />
        ) : (
          <ChessComForm onGames={onGames} onGame={onGame} />
        )}
      </div>

      <details className="mt-6 rounded-md border border-border bg-panel/50 p-3 text-sm text-muted">
        <summary className="cursor-pointer font-medium text-primary">How to read the review</summary>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>
            Each move gets a label: Best, Excellent, Good, Book (opening theory),
            Inaccuracy, Mistake, Blunder, and the rare Great and Brilliant.
          </li>
          <li>
            A move you found in a hard position with little time left is marked in
            amber, on the board and in the move list. That is a strong move played
            under real pressure.
          </li>
          <li>
            Select any move to see why it earned its label, in plain words. Step
            through with the move list, the arrow keys, or the graph.
          </li>
          <li>
            Blitz and rapid games with move times get the time-aware second opinion.
            Other games still get the full normal review.
          </li>
        </ul>
      </details>
    </div>
  );
}

function Tab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`flex-1 rounded px-3 py-1.5 font-medium transition-colors ${
        active ? "bg-accent text-app" : "text-muted hover:text-primary"
      }`}
    >
      {children}
    </button>
  );
}
