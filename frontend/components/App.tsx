"use client";

import { useState } from "react";
import { intake, runAnalysis } from "@/lib/api";
import type {
  AnalysisResult,
  ChessComGame,
  ChessComGamesResponse,
  GameSummary,
  JobProgress,
} from "@/lib/types";
import AnalysisView from "./AnalysisView";
import ChessComPicker from "./ChessComPicker";
import GamePicker from "./GamePicker";
import Landing from "./Landing";
import Progress from "./Progress";

type Phase =
  | { kind: "input" }
  | { kind: "picker"; games: GameSummary[] }
  | { kind: "chesscom"; resp: ChessComGamesResponse; username: string }
  | { kind: "progress" }
  | { kind: "analysis"; result: AnalysisResult };

function friendly(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (/fetch|network|load failed/i.test(msg)) {
    return "Could not reach the review service on this machine. Make sure it is running.";
  }
  return msg;
}

export default function App() {
  const [phase, setPhase] = useState<Phase>({ kind: "input" });
  const [pgn, setPgn] = useState("");
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyze = async (pgnText: string, index: number) => {
    setPgn(pgnText);
    setProgress(null);
    setPhase({ kind: "progress" });
    try {
      const result = await runAnalysis(pgnText, index, setProgress);
      setPhase({ kind: "analysis", result });
    } catch (e) {
      setError(friendly(e));
      setPhase({ kind: "input" });
    }
  };

  // Paste or upload: parse locally, then pick a game or go straight to analysis.
  const submit = async (pgnText: string) => {
    setError(null);
    setPgn(pgnText);
    try {
      const { games } = await intake(pgnText);
      const accepted = games.filter((g) => g.accepted);
      if (accepted.length === 0) {
        setError(games[0]?.reject_reason ?? "No standard game found in that PGN.");
      } else if (games.length === 1) {
        await analyze(pgnText, accepted[0].index);
      } else {
        setPhase({ kind: "picker", games });
      }
    } catch (e) {
      setError(friendly(e));
    }
  };

  const reset = () => {
    setError(null);
    setPhase({ kind: "input" });
  };

  switch (phase.kind) {
    case "input":
      return (
        <Landing
          onPasteSubmit={submit}
          onGames={(resp, username) => setPhase({ kind: "chesscom", resp, username })}
          onGame={(game: ChessComGame) => analyze(game.pgn, 0)}
          pasteError={error}
        />
      );
    case "picker":
      return <GamePicker games={phase.games} onPick={(i) => analyze(pgn, i)} onBack={reset} />;
    case "chesscom":
      return (
        <ChessComPicker
          initial={phase.resp}
          username={phase.username}
          onPick={(p) => analyze(p, 0)}
          onBack={reset}
        />
      );
    case "progress":
      return <Progress progress={progress} />;
    case "analysis":
      return <AnalysisView result={phase.result} onReset={reset} />;
  }
}
