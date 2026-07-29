"use client";

import { useEffect, useRef } from "react";
import type { MoveResult } from "@/lib/types";
import ClassBadge from "./ClassBadge";

interface Props {
  moves: MoveResult[];
  curr: number;
  onSelect: (i: number) => void;
}

interface Cell {
  m: MoveResult;
  i: number;
}
interface Row {
  num: number;
  white?: Cell;
  black?: Cell;
}

export default function MoveList({ moves, curr, onSelect }: Props) {
  const activeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [curr]);

  const rows: Row[] = [];
  moves.forEach((m, i) => {
    let row = rows[rows.length - 1];
    if (!row || row.num !== m.move_number) {
      row = { num: m.move_number };
      rows.push(row);
    }
    if (m.side === "white") row.white = { m, i };
    else row.black = { m, i };
  });

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      {rows.map((row) => (
        <div key={row.num} className="flex items-stretch odd:bg-panel2/20">
          <div className="w-7 shrink-0 px-1 py-1 text-right text-xs text-faint tabular-nums">
            {row.num}.
          </div>
          <MoveCell cell={row.white} curr={curr} onSelect={onSelect} activeRef={activeRef} />
          <MoveCell cell={row.black} curr={curr} onSelect={onSelect} activeRef={activeRef} />
        </div>
      ))}
    </div>
  );
}

function MoveCell({
  cell,
  curr,
  onSelect,
  activeRef,
}: {
  cell?: Cell;
  curr: number;
  onSelect: (i: number) => void;
  activeRef: React.RefObject<HTMLButtonElement | null>;
}) {
  if (!cell) return <div className="flex-1" />;
  const { m, i } = cell;
  const active = curr === i;
  const label =
    `${m.move_number}${m.side === "white" ? "." : "..."}${m.san} ${m.baseline_label}` +
    (m.upgraded ? ` upgraded to ${m.time_aware_label} under time pressure` : "");
  return (
    <button
      ref={active ? activeRef : undefined}
      onClick={() => onSelect(i)}
      aria-label={label}
      aria-current={active ? "true" : undefined}
      className={`flex flex-1 items-center gap-1.5 px-2 py-1 text-left text-sm transition-colors ${
        active ? "bg-panel2 text-primary ring-1 ring-inset ring-border" : "text-muted hover:bg-panel2/60"
      }`}
    >
      <span className="font-medium">{m.san}</span>
      {!m.in_book && <ClassBadge label={m.baseline_label} upgraded={m.upgraded} size="sm" />}
    </button>
  );
}
