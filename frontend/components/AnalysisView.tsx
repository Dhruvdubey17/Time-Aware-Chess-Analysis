"use client";

import { useCallback, useEffect, useState } from "react";
import { uciSquares } from "@/lib/chess";
import type { AnalysisResult, MoveResult } from "@/lib/types";
import Board from "./Board";
import EvalBar from "./EvalBar";
import EvalGraph from "./EvalGraph";
import MoveCallout from "./MoveCallout";
import MoveList from "./MoveList";
import PlayerBar from "./PlayerBar";
import Summary from "./Summary";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Clock a side has left at the current move: the most recent clock reading for
// that color up to here, or the base time before their first move.
function currentClock(
  moves: MoveResult[],
  curr: number,
  color: "white" | "black",
  timeControl: string,
): number | null {
  const upto = curr < 0 ? -1 : Math.min(curr, moves.length - 1);
  for (let i = upto; i >= 0; i--) {
    if (moves[i].side === color && moves[i].clock_after_s != null) return moves[i].clock_after_s;
  }
  if (!moves.some((m) => m.clock_after_s != null)) return null; // game has no clocks
  const base = parseInt(timeControl, 10);
  return Number.isFinite(base) ? base : null;
}

export default function AnalysisView({
  result,
  onReset,
}: {
  result: AnalysisResult;
  onReset: () => void;
}) {
  const moves = result.moves;
  const n = moves.length;
  const [curr, setCurr] = useState(-1);
  const [orientation, setOrientation] = useState<"white" | "black">("white");

  const go = useCallback((i: number) => setCurr(Math.max(-1, Math.min(n - 1, i))), [n]);
  const jumpToPly = useCallback(
    (ply: number) => {
      const idx = moves.findIndex((m) => m.ply === ply);
      if (idx >= 0) setCurr(idx);
    },
    [moves],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") { e.preventDefault(); setCurr((c) => Math.max(-1, c - 1)); }
      else if (e.key === "ArrowRight") { e.preventDefault(); setCurr((c) => Math.min(n - 1, c + 1)); }
      else if (e.key === "ArrowUp" || e.key === "Home") { e.preventDefault(); setCurr(-1); }
      else if (e.key === "ArrowDown" || e.key === "End") { e.preventDefault(); setCurr(n - 1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [n]);

  const fen =
    curr < 0
      ? moves[0]?.fen_before ?? START_FEN
      : curr < n - 1
        ? moves[curr + 1].fen_before
        : result.final_fen || moves[curr].fen_before;
  const cur = curr >= 0 ? moves[curr] : null;
  const cpWhite = cur ? cur.eval_white : 0;
  const dest = cur ? uciSquares(cur.uci).to : null;
  const marker = cur && !cur.in_book && dest
    ? { square: dest, label: cur.baseline_label, upgraded: cur.upgraded }
    : null;

  const bottomColor = orientation;
  const topColor = orientation === "white" ? "black" : "white";
  const nameOf = (c: "white" | "black") => (c === "white" ? result.white : result.black);
  const ratingOf = (c: "white" | "black") => (c === "white" ? result.white_elo : result.black_elo);
  const bar = (c: "white" | "black") => (
    <PlayerBar
      name={nameOf(c)}
      rating={ratingOf(c)}
      color={c}
      clock={currentClock(moves, curr, c, result.time_control)}
    />
  );

  return (
    <div className="mx-auto max-w-[1700px] p-3 sm:p-4">
      <div className="flex flex-col gap-4 lg:h-[calc(100vh-1.5rem)] lg:flex-row lg:items-stretch">
        {/* Board column, the centerpiece: fills the height so the board is big */}
        <div className="flex min-w-0 flex-1 justify-center">
          <div className="flex w-full flex-col lg:h-full lg:max-w-[calc(100vh-5.5rem)]">
            <div className="shrink-0">{bar(topColor)}</div>
            <div className="flex min-h-0 flex-1 gap-1.5">
              <EvalBar cpWhite={cpWhite} orientation={orientation} />
              {/* Largest square that fits the space: width-based on mobile,
                  centered and height-filling on desktop. */}
              <div className="relative min-h-0 min-w-0 flex-1">
                <div className="aspect-square w-full lg:absolute lg:inset-0 lg:m-auto lg:w-auto lg:max-h-full lg:max-w-full">
                  <Board
                    fen={fen}
                    orientation={orientation}
                    lastMoveUci={cur?.uci ?? null}
                    pressureSquare={cur?.upgraded ? dest : null}
                    marker={marker}
                  />
                </div>
              </div>
            </div>
            <div className="shrink-0">{bar(bottomColor)}</div>
          </div>
        </div>

        {/* Review panel, bound to the same height so it does not leak below */}
        <div className="flex min-h-0 flex-col gap-3 lg:h-full lg:w-[360px]">
          <Header result={result} onReset={onReset}
            onFlip={() => setOrientation((o) => (o === "white" ? "black" : "white"))} />

          {curr < 0 ? (
            <Summary result={result} onJump={jumpToPly} />
          ) : (
            <MoveCallout move={cur} timeAwareAvailable={result.time_aware_available}
              note={result.time_aware_note} />
          )}

          <div className="flex max-h-[45vh] min-h-0 flex-1 flex-col rounded-md border border-border bg-panel lg:max-h-none">
            <Nav curr={curr} n={n} go={go} />
            <MoveList moves={moves} curr={curr} onSelect={go} />
          </div>

          <EvalGraph moves={moves} curr={curr} onJump={go} />
        </div>
      </div>
    </div>
  );
}

function Header({ result, onReset, onFlip }: {
  result: AnalysisResult; onReset: () => void; onFlip: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="min-w-0">
        <h1 className="text-base font-semibold">Game review</h1>
        <p className="truncate text-xs text-muted">
          {result.result} · {result.opening || "opening unknown"}
          {result.regime ? ` · ${result.regime} ${result.time_control}` : ""}
        </p>
      </div>
      <div className="flex shrink-0 gap-1.5">
        <button onClick={onFlip} title="Flip board"
          className="rounded border border-border bg-panel px-2.5 py-1 text-xs text-muted hover:text-primary">
          Flip
        </button>
        <button onClick={onReset}
          className="rounded border border-border bg-panel px-2.5 py-1 text-xs text-muted hover:text-primary">
          New game
        </button>
      </div>
    </div>
  );
}

function Nav({ curr, n, go }: { curr: number; n: number; go: (i: number) => void }) {
  const btn = "flex-1 py-1.5 text-muted hover:bg-panel2 hover:text-primary disabled:opacity-30";
  return (
    <div className="flex border-b border-border">
      <button className={btn} onClick={() => go(-1)} disabled={curr < 0} aria-label="First">⏮</button>
      <button className={btn} onClick={() => go(curr - 1)} disabled={curr < 0} aria-label="Previous">◀</button>
      <button className={btn} onClick={() => go(curr + 1)} disabled={curr >= n - 1} aria-label="Next">▶</button>
      <button className={btn} onClick={() => go(n - 1)} disabled={curr >= n - 1} aria-label="Last">⏭</button>
    </div>
  );
}
