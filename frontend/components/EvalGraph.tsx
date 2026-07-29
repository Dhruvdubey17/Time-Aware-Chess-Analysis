"use client";

import { whiteWinPct } from "@/lib/chess";
import { NOTABLE, UPGRADE_COLOR, labelColor } from "@/lib/labels";
import type { MoveResult } from "@/lib/types";

// A compact eval history across the whole game, like the chess.com graph: a
// white area whose height is white's win chance at each move, a midline, colored
// dots where notable moves happened, and a cursor at the current move. Click to
// jump to a point.
export default function EvalGraph({
  moves,
  curr,
  onJump,
}: {
  moves: MoveResult[];
  curr: number;
  onJump: (i: number) => void;
}) {
  const n = moves.length;
  if (n < 2) return null;
  const denom = n - 1;

  const pts = moves.map((m, i) => ({ x: i, y: 100 - whiteWinPct(m.eval_white) }));
  const area =
    `M0,0 L0,${pts[0].y} ` +
    pts.map((p) => `L${p.x},${p.y}`).join(" ") +
    ` L${denom},0 Z`;

  const dots = moves
    .map((m, i) => ({ m, i }))
    .filter(({ m }) => m.upgraded || NOTABLE.has(m.baseline_label));

  const onClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    onJump(Math.round(Math.max(0, Math.min(1, frac)) * denom));
  };

  return (
    <div>
      <div className="mb-1 text-xs text-faint">Evaluation over the game</div>
      <div
        onClick={onClick}
        className="relative h-16 w-full cursor-pointer overflow-hidden rounded bg-[#403e3b]"
      >
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${denom} 100`}
          preserveAspectRatio="none"
        >
          <path d={area} fill="#e9e7df" />
          <line x1="0" y1="50" x2={denom} y2="50" stroke="#00000033" strokeWidth="0.4" />
        </svg>

        {dots.map(({ m, i }) => (
          <span
            key={i}
            className="absolute h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-1 ring-black/30"
            style={{
              left: `${(i / denom) * 100}%`,
              top: `${100 - whiteWinPct(m.eval_white)}%`,
              background: m.upgraded ? UPGRADE_COLOR : labelColor(m.baseline_label),
            }}
          />
        ))}

        {curr >= 0 && (
          <span
            className="absolute top-0 h-full w-px bg-accent"
            style={{ left: `${(curr / denom) * 100}%` }}
          />
        )}
      </div>
    </div>
  );
}
