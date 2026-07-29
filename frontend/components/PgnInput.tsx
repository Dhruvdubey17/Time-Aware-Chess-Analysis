"use client";

import { useRef, useState } from "react";

const SAMPLE = `[Event "Rated Blitz game"]
[White "you"]
[Black "opponent"]
[Result "1-0"]
[TimeControl "180+2"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 ...`;

// The paste/upload panel. Fully offline: it never touches the network.
export default function PgnInput({
  onSubmit,
  error,
}: {
  onSubmit: (pgn: string) => void;
  error?: string | null;
}) {
  const [pgn, setPgn] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) f.text().then(setPgn);
  };

  return (
    <div>
      <textarea
        value={pgn}
        onChange={(e) => setPgn(e.target.value)}
        placeholder={SAMPLE}
        spellCheck={false}
        className="h-56 w-full resize-y rounded-md border border-border bg-panel p-3 font-mono text-sm text-primary placeholder:text-faint focus:border-accent focus:outline-none"
      />

      {error && <p className="mt-2 text-sm text-bad">{error}</p>}

      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={() => onSubmit(pgn)}
          disabled={!pgn.trim()}
          className="rounded-md bg-accent px-5 py-2 font-medium text-app hover:brightness-110 disabled:opacity-40"
        >
          Review game
        </button>
        <button
          onClick={() => fileRef.current?.click()}
          className="rounded-md border border-border px-4 py-2 text-muted hover:text-primary"
        >
          Upload .pgn
        </button>
        <input ref={fileRef} type="file" accept=".pgn,.txt" onChange={onFile} className="hidden" />
      </div>
    </div>
  );
}
