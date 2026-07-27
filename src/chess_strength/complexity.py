"""Phase 5: how hard was the position to play.

Two tiers so the pipeline can move before we spend any engine time.

Tier A is free. It reads the `[%eval]` line the games already carry and turns it
into proxies for sharpness: how much the eval swings over the next few plies,
how noisy it is locally, and whether the game is still in the balance. This lets
Phases 7 and 8 run with zero Stockfish.

Tier B is the real thing. It asks Stockfish for the top few moves at a fixed node
budget and measures how close the alternatives are: the gap between the best two,
how many moves are reasonable (a bar that tightens as rating rises), how spread
out the choice is (entropy), and whether there was really only one move. It is
meant to run on a sample of positions, not every one, and it caches by FEN.

Tier A works on parse_moves rows. Evals there come in two columns: eval_cp_white
is the continuous White-POV line (use it for swings and volatility so the number
does not flip sign every ply), and eval_cp is the mover-POV value (use it to say
who stands better).
"""

from __future__ import annotations

import math
from statistics import pstdev

# Tier A windows, in plies.
SWING_PLIES = 4       # how far ahead to look for the eval to move
VOL_PLIES = 6         # local window for eval volatility

# A game is "decided" for the side to move past this many centipawns.
DECISIVE_CP = 200

# Softmax temperature (cp) for turning candidate evals into a choice
# distribution. Roughly a pawn's worth of spread.
SOFTMAX_TEMP_CP = 100.0

# Mate stored as a large signed cp, matching parse_moves.
_MATE_CP = 10000


# ---------------------------------------------------------------------------
# Tier A: zero-compute proxies from the eval line
# ---------------------------------------------------------------------------

def eval_swing_next(white_evals: list[int], i: int, plies: int = SWING_PLIES) -> float:
    """Largest absolute move of the White-POV eval over the next `plies`.

    A big swing just after this position means small changes in play flipped the
    assessment a lot, so the position was sharp. Zero if nothing follows.
    """
    here = white_evals[i]
    ahead = white_evals[i + 1 : i + 1 + plies]
    if not ahead:
        return 0.0
    return float(max(abs(e - here) for e in ahead))


def eval_volatility(white_evals: list[int], i: int, window: int = VOL_PLIES) -> float:
    """Std of the White-POV eval in a window centred on ply i.

    Local noise in the eval, another read on how unsettled the position is.
    Zero when there is only one point in range.
    """
    half = window // 2
    lo = max(0, i - half)
    hi = min(len(white_evals), i + half + 1)
    chunk = white_evals[lo:hi]
    if len(chunk) < 2:
        return 0.0
    return float(pstdev(chunk))


def decisiveness_bucket(mover_cp: int) -> str:
    """Coarse state of the game from the mover's POV: winning, equal, or losing."""
    if mover_cp > DECISIVE_CP:
        return "winning"
    if mover_cp < -DECISIVE_CP:
        return "losing"
    return "equal"


def add_tier_a(rows: list[dict]) -> list[dict]:
    """Add Tier-A complexity fields to per-move rows from parse_moves.

    Rows are grouped by game and must be in ply order (as parse_moves emits
    them). Adds `eval_swing`, `eval_volatility`, and `decisiveness`.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["game_id"], []).append(row)

    for group in groups.values():
        white_evals = [r["eval_cp_white"] for r in group]
        for i, row in enumerate(group):
            row["eval_swing"] = eval_swing_next(white_evals, i)
            row["eval_volatility"] = eval_volatility(white_evals, i)
            row["decisiveness"] = decisiveness_bucket(row["eval_cp"])

    return rows


# ---------------------------------------------------------------------------
# Tier B: Stockfish MultiPV on a sample
# ---------------------------------------------------------------------------

def reasonable_threshold_cp(rating: int | None) -> int:
    """How close a move must be to the best to count as reasonable.

    Weaker players have more moves that are "fine", so the bar is looser: ~100 cp
    at 1400 and below, tightening to ~10 cp at 2400 and above, linear between.
    Unknown rating falls back to the loose end.
    """
    if rating is None or rating <= 1400:
        return 100
    if rating >= 2400:
        return 10
    frac = (rating - 1400) / (2400 - 1400)
    return round(100 - frac * (100 - 10))


def decision_entropy(cps: list[int], temp: float = SOFTMAX_TEMP_CP) -> float:
    """Normalized entropy of the softmax over candidate evals, in [0, 1].

    Near 0 when one move dominates, near 1 when several are equally good. Values
    are shifted by the max first for numerical stability.
    """
    if len(cps) < 2:
        return 0.0
    top = max(cps)
    weights = [math.exp((c - top) / temp) for c in cps]
    total = sum(weights)
    probs = [w / total for w in weights]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy / math.log(len(cps))


def complexity_features(
    cps: list[int], legal_move_count: int, rating: int | None = None
) -> dict:
    """Tier-B features from a sorted (best-first) list of candidate evals.

    `cps` are side-to-move centipawns for the top moves Stockfish returned.
    `legal_move_count` is the number of legal moves in the position.
    """
    if not cps:
        raise ValueError("need at least one candidate eval")

    best = cps[0]
    thr = reasonable_threshold_cp(rating)
    n_reasonable = sum(1 for c in cps if c >= best - thr)
    return {
        "eval_gap_1_2": (best - cps[1]) if len(cps) > 1 else None,
        "n_reasonable": n_reasonable,
        "decision_entropy": decision_entropy(cps),
        "is_only_move": legal_move_count == 1 or n_reasonable <= 1,
    }


class FenAnalyzer:
    """Runs Stockfish MultiPV on a position and caches results by FEN.

    Same FEN twice hits the cache instead of the engine, which matters because a
    single position can repeat across games (transpositions, common openings).
    `hits` counts cache hits so callers can see the saving.
    """

    def __init__(self, engine, nodes: int, multipv: int = 4):
        self._engine = engine
        self._nodes = nodes
        self._multipv = multipv
        self._cache: dict[str, list[int]] = {}
        self.hits = 0

    def candidate_cps(self, fen: str) -> list[int]:
        """Side-to-move centipawns for the top moves, best first."""
        if fen in self._cache:
            self.hits += 1
            return self._cache[fen]

        import chess
        import chess.engine

        board = chess.Board(fen)
        infos = self._engine.analyse(
            board, chess.engine.Limit(nodes=self._nodes), multipv=self._multipv
        )
        cps = sorted(
            (info["score"].relative.score(mate_score=_MATE_CP) for info in infos),
            reverse=True,
        )
        self._cache[fen] = cps
        return cps
