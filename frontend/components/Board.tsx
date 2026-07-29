"use client";

import type { CSSProperties } from "react";
import { Chessboard } from "react-chessboard";
import { uciSquares } from "@/lib/chess";
import ClassBadge from "./ClassBadge";

interface Marker {
  square: string;
  label: string;
  upgraded: boolean;
}

interface Props {
  fen: string;
  orientation: "white" | "black";
  lastMoveUci?: string | null;
  pressureSquare?: string | null;
  marker?: Marker | null;
}

const LAST_MOVE: CSSProperties = { background: "rgba(246, 229, 141, 0.45)" };
const PRESSURE: CSSProperties = { background: "rgba(224, 168, 58, 0.5)" };

// Top-right corner of a square, in board percentages, respecting orientation.
function corner(square: string, orientation: "white" | "black") {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  const col = orientation === "white" ? file : 7 - file;
  const rowFromTop = orientation === "white" ? 7 - rank : rank;
  return { left: `${(col + 1) * 12.5}%`, top: `${rowFromTop * 12.5}%` };
}

export default function Board({ fen, orientation, lastMoveUci, pressureSquare, marker }: Props) {
  const squareStyles: Record<string, CSSProperties> = {};
  if (lastMoveUci) {
    const { from, to } = uciSquares(lastMoveUci);
    squareStyles[from] = LAST_MOVE;
    squareStyles[to] = LAST_MOVE;
  }
  if (pressureSquare) squareStyles[pressureSquare] = PRESSURE;

  return (
    <div className="relative h-full w-full overflow-hidden rounded-md">
      <Chessboard
        options={{
          id: "review-board",
          position: fen,
          boardOrientation: orientation,
          allowDragging: false,
          showNotation: true,
          animationDurationInMs: 150,
          squareStyles,
          darkSquareStyle: { backgroundColor: "#6f9350" },
          lightSquareStyle: { backgroundColor: "#eeeed2" },
          darkSquareNotationStyle: { color: "rgba(238,238,210,0.7)" },
          lightSquareNotationStyle: { color: "rgba(111,147,80,0.85)" },
        }}
      />
      {marker && (
        <div
          className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
          style={corner(marker.square, orientation)}
        >
          <ClassBadge label={marker.label} upgraded={marker.upgraded} size="md" />
        </div>
      )}
    </div>
  );
}
