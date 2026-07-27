"""Phase 7: phase- and complexity-adjusted residual skill score.

The cheap, interpretable core of the idea, and the go/no-go gate. We learn the
normal win-percent loss (WPL) for a situation and score a player by how much
better than that they played. Play better than expected and the residual is
positive.

Revised after the first run. The situation is defined by exogenous difficulty
only: the game phase and how sharp the position the player was handed was, read
from the engine before the move. We do NOT condition on eval bucket, clock
usage, or eval volatility. Those are consequences of skill, so subtracting an
expected WPL built from them would subtract the very signal we want. Only the
engine's pre-move view of the position is exogenous. See PLAN Phase 7 and
standing rule 14.

The engine is used to measure position sharpness, never to grade the move. Its
node budget is fixed and coarse, because we only need to see how many moves are
reasonable, not a precise eval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The situation cell. Exogenous difficulty only, no mediators of skill.
CELL_KEYS = ["game_phase", "complexity_bucket"]

# Fixed, coarse node budget for the complexity read. Fixed so runs reproduce;
# far smaller than the grading budget since ranking the top few moves for a
# sharpness bucket is undemanding. About 25 positions per second here.
COMPLEXITY_NODES = 30000


def tercile(values: pd.Series) -> pd.Series:
    """Split into equal-size low/mid/high, robust to heavy ties.

    Ranking first keeps repeated values from collapsing the bin edges. Falls
    back to a single bin when there is too little to split.
    """
    if values.nunique() < 3 or len(values) < 3:
        return pd.Series(["mid"] * len(values), index=values.index)
    ranks = values.rank(method="first")
    return pd.qcut(ranks, 3, labels=["low", "mid", "high"]).astype(str)


def add_complexity_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket the per-move engine complexity into low/mid/high terciles.

    Expects a `complexity` column (see position_complexity). game_phase already
    comes from Phase 6, so together they form the situation cell.
    """
    df = df.copy()
    df["complexity_bucket"] = tercile(df["complexity"])
    return df


def fit_expected_wpl(
    train: pd.DataFrame, cell_keys: list[str] = CELL_KEYS, alpha: float = 50.0
) -> tuple[dict, float]:
    """Expected WPL per situation cell, shrunk toward the global mean.

    A cell seen only a few times is pulled toward the overall mean so sparse
    cells do not give wild expectations. alpha is the pull strength, in
    effective observations.
    """
    global_mean = float(train["wpl"].mean())
    stats = train.groupby(cell_keys)["wpl"].agg(["mean", "size"])
    shrunk = (stats["size"] * stats["mean"] + alpha * global_mean) / (stats["size"] + alpha)
    return shrunk.to_dict(), global_mean


def add_residuals(
    df: pd.DataFrame, table: dict, global_mean: float, cell_keys: list[str] = CELL_KEYS
) -> pd.DataFrame:
    """Attach expected WPL and the residual (expected minus observed).

    Positive residual means the move lost less win% than the situation expected.
    Unseen cells fall back to the global mean.
    """
    df = df.copy()
    keys = list(df[cell_keys].itertuples(index=False, name=None))
    df["expected_wpl"] = [table.get(k, global_mean) for k in keys]
    df["residual"] = df["expected_wpl"] - df["wpl"]
    return df


def player_scores(df: pd.DataFrame, min_moves: int = 100) -> pd.DataFrame:
    """Per-player skill score (mean residual) and consistency (residual spread).

    Glicko is carried along only as the validation target. Players below
    min_moves are dropped, their scores are too noisy to trust.
    """
    scores = df.groupby("player_id").agg(
        skill_score=("residual", "mean"),
        consistency=("residual", "std"),
        n_moves=("residual", "size"),
        glicko=("player_rating", "mean"),
    )
    scores["consistency"] = scores["consistency"].fillna(0.0)
    return scores[scores["n_moves"] >= min_moves].reset_index()


def raw_wpl_scores(df: pd.DataFrame, min_moves: int = 100) -> pd.DataFrame:
    """Reference baseline: skill score is just negative mean WPL, no conditioning.

    The chosen estimator must at least match this. If it cannot beat plain WPL,
    prefer the simpler score.
    """
    scores = df.groupby("player_id").agg(
        skill_score=("wpl", lambda x: -x.mean()),
        n_moves=("wpl", "size"),
        glicko=("player_rating", "mean"),
    )
    return scores[scores["n_moves"] >= min_moves].reset_index()


def train_test_players(
    df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows so no player appears on both sides. Splits by player, not move."""
    players = df["player_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(players)) < test_frac
    test_ids = set(players[is_test])
    in_test = df["player_id"].isin(test_ids)
    return df[~in_test], df[in_test]


def position_complexity(fens, engine, nodes: int = COMPLEXITY_NODES, multipv: int = 4) -> dict:
    """Engine sharpness per unique FEN, as the entropy over candidate move evals.

    Higher entropy means several moves are close, a more complex choice. Read
    from the position before the move, so it is exogenous to what was played.
    Cached by FEN so repeats are free.
    """
    from .complexity import FenAnalyzer, decision_entropy

    analyzer = FenAnalyzer(engine, nodes=nodes, multipv=multipv)
    return {fen: decision_entropy(analyzer.candidate_cps(fen)) for fen in fens}
