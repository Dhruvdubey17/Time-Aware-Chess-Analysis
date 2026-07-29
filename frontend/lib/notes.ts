import type { MoveResult } from "./types";
import { clockStr } from "./chess";

// The coach-bubble text for a move: a one-line header and a plain sentence.
// Position-specific prose would need an engine explanation we do not have, so
// these are honest, general descriptions of each tier, plus a real explanation
// for our own time-aware upgrades. Human difficulty stays in words.

const PHRASE: Record<string, string> = {
  Book: "a book move",
  Best: "the best move",
  Great: "a great move",
  Excellent: "an excellent move",
  Good: "a good move",
  Inaccuracy: "an inaccuracy",
  Mistake: "a mistake",
  Blunder: "a blunder",
  Brilliant: "a brilliant move",
};

const SENTENCE: Record<string, string> = {
  Book: "Still following opening theory.",
  Best: "The strongest move in the position.",
  Great: "The clear best move, and the alternatives were much worse.",
  Excellent: "A very strong move, almost the best.",
  Good: "A solid, reasonable move.",
  Inaccuracy: "A better move was available.",
  Mistake: "This gives up some of the advantage.",
  Blunder: "This loses a lot. There was a much stronger move.",
  Brilliant: "A daring move that gives up material for a bigger gain.",
};

export function moveNote(move: MoveResult): { header: string; sentence: string } {
  if (move.upgraded) {
    const diff = move.difficulty ?? "hard";
    const clock = move.clock_before_s != null ? ` with ${clockStr(move.clock_before_s)} left` : "";
    return {
      header: `${move.san} was found under pressure`,
      sentence:
        `A ${diff} position${clock}, and a strong move played fast. ` +
        `The time-aware review lifts it from ${move.baseline_label} to ${move.time_aware_label}.`,
    };
  }
  const label = move.baseline_label;
  return {
    header: `${move.san} is ${PHRASE[label] ?? "a move"}`,
    sentence: SENTENCE[label] ?? "",
  };
}
