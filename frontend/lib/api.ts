import type {
  AnalysisResult,
  ChessComGame,
  ChessComGamesResponse,
  GameSummary,
  JobProgress,
} from "./types";

// The backend runs on the same machine. In dev it is a separate port; the Phase
// 5 launcher can point this wherever it serves the API.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data.detail ?? `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}

export function intake(pgn: string): Promise<{ games: GameSummary[] }> {
  return post("/api/intake", { pgn });
}

// The only network-touching calls. They fetch a game from chess.com's public API.
export function chesscomGames(username: string, month?: string): Promise<ChessComGamesResponse> {
  return post("/api/chesscom/games", { username, month });
}

export function chesscomGame(username: string, link: string): Promise<{ game: ChessComGame }> {
  return post("/api/chesscom/game", { username, link });
}

export function startAnalyze(pgn: string, game: number, option = "B"): Promise<{ job_id: string }> {
  return post("/api/analyze", { pgn, game, option });
}

export function getProgress(jobId: string): Promise<JobProgress> {
  return get(`/api/progress/${jobId}`);
}

export function getResult(jobId: string): Promise<AnalysisResult> {
  return get(`/api/result/${jobId}`);
}

// Poll progress until the job finishes, reporting each tick, then fetch the result.
export async function runAnalysis(
  pgn: string,
  game: number,
  onProgress: (p: JobProgress) => void,
  option = "B",
): Promise<AnalysisResult> {
  const { job_id } = await startAnalyze(pgn, game, option);
  for (;;) {
    const p = await getProgress(job_id);
    onProgress(p);
    if (p.status === "error") throw new Error(p.error ?? "Analysis failed.");
    if (p.status === "done") break;
    await new Promise((r) => setTimeout(r, 500));
  }
  return getResult(job_id);
}
