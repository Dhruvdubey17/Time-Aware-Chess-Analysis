"use client";

import type { JobProgress } from "@/lib/types";

const STAGE_TEXT: Record<string, string> = {
  starting: "Getting ready",
  engine: "Studying the positions with the engine",
  maia: "Measuring how hard each move was for a human",
  done: "Finishing up",
};

export default function Progress({ progress }: { progress: JobProgress | null }) {
  const pct = Math.round((progress?.fraction ?? 0) * 100);
  const stage = STAGE_TEXT[progress?.stage ?? "starting"] ?? "Working";

  return (
    <div className="mx-auto flex min-h-full max-w-lg flex-col justify-center p-6">
      <h1 className="text-xl font-semibold">Reviewing your game</h1>
      <p className="mt-1 text-muted">
        This runs entirely on your machine, so a fresh game can take from a few
        seconds to a few minutes. Games you have looked at before are instant.
      </p>

      <div className="mt-6 h-2 w-full overflow-hidden rounded-full bg-panel2">
        <div
          className="h-full rounded-full bg-accent transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-2 text-sm text-muted">
        {stage}
        {progress && progress.total > 0 ? ` (${progress.done}/${progress.total})` : ""} · {pct}%
      </p>
    </div>
  );
}
