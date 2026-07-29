// Small pure helpers for the board and eval display. The Win% sigmoid mirrors
// the backend (Lichess constant) so the eval bar reads the same as the labels.

const WINPROB_K = 0.00368208;
const MATE_FLOOR = 9000;

// White's win chance in [0, 100] for a White-POV centipawn eval.
export function whiteWinPct(cpWhite: number): number {
  if (cpWhite >= MATE_FLOOR) return 100;
  if (cpWhite <= -MATE_FLOOR) return 0;
  const c = Math.max(-1000, Math.min(1000, cpWhite));
  return 50 + 50 * (2 / (1 + Math.exp(-WINPROB_K * c)) - 1);
}

// Human-readable eval from White's POV: "+1.2", "-0.4", "M3", "-M2".
export function formatEval(cpWhite: number): string {
  if (cpWhite >= MATE_FLOOR) return `M${10000 - cpWhite}`;
  if (cpWhite <= -MATE_FLOOR) return `-M${10000 + cpWhite}`;
  const p = cpWhite / 100;
  return (p >= 0 ? "+" : "") + p.toFixed(1);
}

export function uciSquares(uci: string): { from: string; to: string } {
  return { from: uci.slice(0, 2), to: uci.slice(2, 4) };
}

// Seconds to a compact clock string: 65 -> "1:05", 8 -> "0:08".
export function clockStr(sec: number | null): string {
  if (sec == null) return "-";
  const s = Math.round(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
