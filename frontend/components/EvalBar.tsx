"use client";

import { formatEval, whiteWinPct } from "@/lib/chess";

// A thin bar hugging the left edge of the board, as tall as the board, with the
// numeric eval above it. White's share fills from white's side and flips with
// the board.
export default function EvalBar({
  cpWhite,
  orientation,
}: {
  cpWhite: number;
  orientation: "white" | "black";
}) {
  const white = whiteWinPct(cpWhite);
  const whiteBottom = orientation === "white";
  const whiteDiv = <div style={{ height: `${white}%`, background: "#f4f4ee" }} />;
  const blackDiv = <div style={{ height: `${100 - white}%`, background: "#403e3b" }} />;

  return (
    <div className="flex w-7 shrink-0 flex-col items-center self-stretch">
      <span className="mb-1 h-4 text-[11px] font-semibold tabular-nums text-muted">
        {formatEval(cpWhite)}
      </span>
      <div className="flex w-3.5 flex-1 flex-col overflow-hidden rounded-full">
        {whiteBottom ? blackDiv : whiteDiv}
        {whiteBottom ? whiteDiv : blackDiv}
      </div>
    </div>
  );
}
