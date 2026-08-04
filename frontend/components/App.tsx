"use client";

import { useEffect, useState } from "react";
import { chesscomGames, getLockedUser, intake, runAnalysis } from "@/lib/api";
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
  | { kind: "loading" }
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
  const [phase, setPhase] = useState<Phase>({ kind: "loading" });
  const [pgn, setPgn] = useState("");
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Set when the launcher was started with a chess.com username. It makes the
  // games page the app's home: we open there and go back there, not the landing
  // screen.
  const [locked, setLocked] = useState<{ user: string; resp: ChessComGamesResponse } | null>(null);

  // On load, ask the server whether it was launched locked to a user. If so, open
  // straight to that account's games. A refresh reruns this, so a locked launch
  // always lands on the games page.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const user = await getLockedUser();
        if (!user) {
          if (!cancelled) setPhase({ kind: "input" });
          return;
        }
        const resp = await chesscomGames(user);
        if (cancelled) return;
        setLocked({ user, resp });
        setPhase({ kind: "chesscom", resp, username: user });
      } catch (e) {
        // Bad username or a fetch problem: fall back to the normal landing so the
        // app still works, and show what went wrong.
        if (cancelled) return;
        setError(friendly(e));
        setPhase({ kind: "input" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const analyze = async (pgnText: string, index: number) => {
    setPgn(pgnText);
    setProgress(null);
    setPhase({ kind: "progress" });
    try {
      const result = await runAnalysis(pgnText, index, setProgress);
      setPhase({ kind: "analysis", result });
    } catch (e) {
      setError(friendly(e));
      back();
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

  // Home. For a locked launch that is the account's games page, otherwise the
  // landing screen.
  const back = () => {
    setError(null);
    if (locked) setPhase({ kind: "chesscom", resp: locked.resp, username: locked.user });
    else setPhase({ kind: "input" });
  };

  switch (phase.kind) {
    case "loading":
      return (
        <div className="mx-auto flex min-h-full max-w-2xl items-center justify-center p-6 text-muted">
          Loading…
        </div>
      );
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
      return <GamePicker games={phase.games} onPick={(i) => analyze(pgn, i)} onBack={back} />;
    case "chesscom":
      return (
        <ChessComPicker
          initial={phase.resp}
          username={phase.username}
          locked={locked !== null}
          onPick={(p) => analyze(p, 0)}
          onBack={back}
        />
      );
    case "progress":
      return <Progress progress={progress} />;
    case "analysis":
      return <AnalysisView result={phase.result} onReset={back} />;
  }
}
