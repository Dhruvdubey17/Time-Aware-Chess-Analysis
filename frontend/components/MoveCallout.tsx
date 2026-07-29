"use client";

import { clockStr, formatEval } from "@/lib/chess";
import { moveNote } from "@/lib/notes";
import type { MoveResult } from "@/lib/types";
import ClassBadge from "./ClassBadge";

// The coach bubble for the current move: the badge, a one-line header with the
// eval, a plain sentence, and the supporting facts. When a move was upgraded
// under pressure the whole card takes the amber treatment.
export default function MoveCallout({
  move,
  timeAwareAvailable,
  note,
}: {
  move: MoveResult | null;
  timeAwareAvailable: boolean;
  note: string;
}) {
  if (!move) {
    return (
      <div className="rounded-md border border-border bg-panel p-3 text-sm text-muted">
        Step through the game to see how each move was judged.
      </div>
    );
  }

  const n = moveNote(move);
  const up = move.upgraded;

  return (
    <div className={`rounded-md border p-3 ${up ? "border-accent bg-accent/10" : "border-border bg-panel"}`}>
      <div className="flex items-start gap-2.5">
        <ClassBadge label={move.baseline_label} upgraded={up} size="md" />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <span className="font-semibold">{n.header}</span>
            <span className="ml-auto shrink-0 text-sm tabular-nums text-muted">
              {formatEval(move.eval_white)}
            </span>
          </div>
          <p className={`mt-0.5 text-sm ${up ? "text-accent" : "text-muted"}`}>{n.sentence}</p>
        </div>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
        <Fact k="Win% lost" v={move.wpl.toFixed(1)} />
        {move.difficulty && <Fact k="Hard for a human?" v={cap(move.difficulty)} />}
        {move.clock_before_s != null && <Fact k="Clock left" v={clockStr(move.clock_before_s)} />}
        {move.time_spent_s != null && move.expected_think_s != null && (
          <Fact
            k="Time on move"
            v={`${move.time_spent_s.toFixed(0)}s (usually ~${move.expected_think_s.toFixed(0)}s)`}
          />
        )}
      </dl>

      {!timeAwareAvailable && <p className="mt-2 text-xs text-muted">{note}</p>}
    </div>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="text-faint">{k}</dt>
      <dd className="tabular-nums text-primary">{v}</dd>
    </div>
  );
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
