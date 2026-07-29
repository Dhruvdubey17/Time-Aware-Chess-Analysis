// Shapes returned by the local backend. Kept in step with backend/intake.py
// (GameReport.summary) and backend/analyze.py (AnalysisResult, MoveResult).

export type Label =
  | "Brilliant" | "Great" | "Best" | "Excellent" | "Good"
  | "Book" | "Inaccuracy" | "Mistake" | "Blunder";

export interface GameSummary {
  index: number;
  accepted: boolean;
  reject_reason: string | null;
  white: string;
  black: string;
  white_elo: number | null;
  black_elo: number | null;
  result: string;
  termination: string;
  opening: string;
  site: string;
  time_control: string;
  regime: string | null;
  has_clocks: boolean;
  has_evals: boolean;
  n_plies: number;
  n_moves: number;
  time_aware_available: boolean;
  capability_note: string;
}

export interface MoveResult {
  ply: number;
  move_number: number;
  side: "white" | "black";
  san: string;
  uci: string;
  fen_before: string;
  phase: string;
  in_book: boolean;
  win_before: number;
  win_after: number;
  wpl: number;
  eval_white: number;
  sac_cp: number;
  clock_before_s: number | null;
  clock_after_s: number | null;
  time_spent_s: number | null;
  expected_think_s: number | null;
  residual_s: number | null;
  maia_entropy: number | null;
  difficulty: string | null;
  gate: number | null;
  pressure: number | null;
  skill: number | null;
  baseline_label: Label;
  time_aware_label: Label | null;
  upgraded: boolean;
}

export interface Upgrade {
  ply: number;
  move_number: number;
  side: "white" | "black";
  san: string;
  baseline: Label;
  time_aware: Label;
  wpl: number;
  maia_entropy: number | null;
  difficulty: string | null;
  time_spent_s: number | null;
  expected_think_s: number | null;
  clock_before_s: number | null;
  fen_before: string;
}

export interface Summary {
  n_moves: number;
  baseline_counts: Record<string, number>;
  time_aware_counts: Record<string, number> | null;
  n_upgrades: number;
  upgrades: Upgrade[];
}

export interface AnalysisResult {
  white: string;
  black: string;
  white_elo: number | null;
  black_elo: number | null;
  result: string;
  opening: string;
  site: string;
  regime: string | null;
  time_control: string;
  final_fen: string;
  time_aware_available: boolean;
  time_aware_note: string;
  moves: MoveResult[];
  summary: Summary;
}

export interface ChessComGame {
  white: string;
  white_elo: number | null;
  black: string;
  black_elo: number | null;
  result: string;
  time_class: string;
  time_control: string;
  date: string;
  url: string;
  rules: string;
  pgn: string;
}

export interface ChessComGamesResponse {
  month: string;
  months: string[];
  games: ChessComGame[];
}

export interface JobProgress {
  status: "running" | "done" | "error";
  stage: string;
  done: number;
  total: number;
  fraction: number;
  error: string | null;
}
