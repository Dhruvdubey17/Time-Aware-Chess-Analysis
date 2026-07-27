"""Phase 7 tests: the revised residual mechanics, offline on synthetic rows.

The estimator now conditions on game phase and engine position complexity only.
Unit tests inject a complexity column so no engine is needed; the real engine
read is a separate engine-marked test. The go/no-go correlation gate needs a
real sample of players and is run by scripts/baseline_residual.py.
"""

import numpy as np
import pandas as pd
import pytest

from chess_strength.baseline_residual import (
    add_complexity_bucket,
    add_residuals,
    fit_expected_wpl,
    player_scores,
    raw_wpl_scores,
    tercile,
    train_test_players,
)
from chess_strength.config import load_config


def _frame(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "player_id": [f"p{i % 30}" for i in range(n)],
            "wpl": rng.uniform(0, 20, n),
            "complexity": rng.uniform(0, 1, n),
            "game_phase": rng.choice(["opening", "middlegame", "endgame"], n),
            "player_rating": rng.integers(1000, 2200, n),
            "regime": rng.choice(["blitz", "rapid"], n),
        }
    )


def test_tercile_splits_three_ways():
    s = pd.Series(range(30))
    assert set(tercile(s)) == {"low", "mid", "high"}
    # Degenerate input falls back to a single bin, no crash.
    assert set(tercile(pd.Series([5, 5, 5]))) == {"mid"}


def test_only_exogenous_axes_in_cell():
    # Guard against a mediator sneaking back into the situation cell.
    from chess_strength.baseline_residual import CELL_KEYS

    assert set(CELL_KEYS) == {"game_phase", "complexity_bucket"}
    for banned in ("eval_bucket", "time_bucket", "eval_volatility", "clock_remaining_s"):
        assert banned not in CELL_KEYS


def test_sparse_cell_shrinks_to_global_mean():
    df = add_complexity_bucket(_frame())
    table, _global_mean = fit_expected_wpl(df, alpha=50.0)
    assert max(table.values()) - min(table.values()) < df["wpl"].std()


def test_residual_sign_and_fallback():
    df = add_complexity_bucket(_frame())
    table, global_mean = fit_expected_wpl(df)
    scored = add_residuals(df, table, global_mean)
    row = scored.iloc[0]
    assert row["residual"] == row["expected_wpl"] - row["wpl"]
    # An unseen cell uses the global mean as the expectation.
    unseen = pd.DataFrame(
        [{"game_phase": "zzz", "complexity_bucket": "low", "wpl": 5.0}]
    )
    out = add_residuals(unseen, table, global_mean)
    assert out["expected_wpl"].iloc[0] == global_mean


def test_player_scores_mean_residual():
    df = add_complexity_bucket(_frame())
    table, global_mean = fit_expected_wpl(df)
    scored = add_residuals(df, table, global_mean)
    players = player_scores(scored, min_moves=1)
    p0 = scored[scored["player_id"] == "p0"]["residual"].mean()
    got = players[players["player_id"] == "p0"]["skill_score"].iloc[0]
    assert abs(got - p0) < 1e-9


def test_raw_wpl_baseline_is_negative_mean():
    df = _frame()
    players = raw_wpl_scores(df, min_moves=1)
    p0_wpl = df[df["player_id"] == "p0"]["wpl"].mean()
    got = players[players["player_id"] == "p0"]["skill_score"].iloc[0]
    assert abs(got - (-p0_wpl)) < 1e-9


def test_train_test_players_disjoint():
    df = _frame()
    train, test = train_test_players(df, test_frac=0.3, seed=0)
    assert set(train["player_id"]) & set(test["player_id"]) == set()
    assert set(train["player_id"]) | set(test["player_id"]) == set(df["player_id"])


@pytest.mark.engine
def test_position_complexity_reads_engine():
    import shutil
    from pathlib import Path

    import chess.engine

    from chess_strength.baseline_residual import position_complexity

    path = load_config()["stockfish_path"]
    if not shutil.which(path) and not Path(path).exists():
        pytest.skip("Stockfish binary not installed")

    # A quiet opening and a sharp middlegame; both should return a real number.
    fens = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 1",
    ]
    with chess.engine.SimpleEngine.popen_uci(path) as engine:
        comp = position_complexity(fens, engine, nodes=20000, multipv=4)
    assert len(comp) == 2
    assert all(0.0 <= v <= 1.0 for v in comp.values())
