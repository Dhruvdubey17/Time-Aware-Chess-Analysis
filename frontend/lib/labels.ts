import type { Label } from "./types";

// Our own clean take on the chess.com tiers: a color and a short symbol per
// classification. Not their proprietary art. Brilliant gets the brightest,
// rarest accent; the mistakes warm up from yellow to red.
const COLORS: Record<Label, string> = {
  Brilliant: "#21c7b8",
  Great: "#66b03f",
  Best: "#81b64c",
  Excellent: "#9fc873",
  Good: "#a6a69c",
  Book: "#b58863",
  Inaccuracy: "#f2c14e",
  Mistake: "#e58f2a",
  Blunder: "#ca3431",
};

const SYMBOLS: Record<Label, string> = {
  Brilliant: "!!",
  Great: "!",
  Best: "★",
  Excellent: "✓",
  Good: "✓",
  Book: "◉",
  Inaccuracy: "?!",
  Mistake: "?",
  Blunder: "??",
};

// The pressure upgrade keeps its own amber identity, separate from the tiers.
export const UPGRADE_COLOR = "#e0a83a";

// Worst to best, for ranking and summary ordering.
export const LADDER: Label[] = [
  "Blunder", "Mistake", "Inaccuracy", "Book", "Good", "Excellent", "Best", "Great", "Brilliant",
];

// Which classifications get a dot on the eval graph (the ones worth jumping to).
export const NOTABLE: ReadonlySet<string> = new Set([
  "Brilliant", "Great", "Inaccuracy", "Mistake", "Blunder",
]);

export function labelColor(label: string | null | undefined): string {
  return (label && COLORS[label as Label]) || "#8b8781";
}

export function labelSymbol(label: string | null | undefined): string {
  return (label && SYMBOLS[label as Label]) || "";
}
