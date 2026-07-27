"""Phase 5 tests: position complexity.

Tier A is checked offline on the fixture and on hand-built eval lines. The pure
Tier-B feature math is checked on synthetic candidate lists. The engine round
trip (real Stockfish, FEN cache, only-move detection) is marked `engine` and
skipped when the binary is absent.
"""

from pathlib import Path

import pytest

from chess_strength.complexity import (
    add_tier_a,
    complexity_features,
    decision_entropy,
    decisiveness_bucket,
    eval_swing_next,
    eval_volatility,
    reasonable_threshold_cp,
)
from chess_strength.config import load_config
from chess_strength.parse_moves import parse_pgn_file

FIXTURE = Path(__file__).parent / "fixtures" / "mini.pgn"


# --- Tier A ---------------------------------------------------------------

def test_eval_swing_next():
    evals = [10, 10, 200, 10]
    # From ply 0, the eval jumps to 200 within the window: swing 190.
    assert eval_swing_next(evals, 0) == 190.0
    # Nothing follows the last ply.
    assert eval_swing_next(evals, 3) == 0.0


def test_eval_volatility_zero_when_flat():
    assert eval_volatility([50, 50, 50, 50], 1) == 0.0
    assert eval_volatility([0, 100], 0) > 0.0


def test_decisiveness_bucket():
    assert decisiveness_bucket(0) == "equal"
    assert decisiveness_bucket(500) == "winning"
    assert decisiveness_bucket(-500) == "losing"
    # On the boundary stays equal.
    assert decisiveness_bucket(200) == "equal"


def test_add_tier_a_on_fixture():
    rows = add_tier_a(list(parse_pgn_file(FIXTURE)))
    for r in rows:
        assert "eval_swing" in r
        assert "eval_volatility" in r
        assert r["decisiveness"] in ("winning", "equal", "losing")
    # Game bbbb2222 sacrifices a knight, so its eval line swings hard.
    g2 = [r for r in rows if r["game_id"] == "bbbb2222"]
    assert max(r["eval_swing"] for r in g2) >= 150.0


# --- Tier B pure math -----------------------------------------------------

def test_reasonable_threshold_taper():
    assert reasonable_threshold_cp(1200) == 100
    assert reasonable_threshold_cp(2600) == 10
    assert reasonable_threshold_cp(None) == 100
    mid = reasonable_threshold_cp(1900)
    assert 10 < mid < 100


def test_decision_entropy_bounds():
    # One dominant move: near zero.
    assert decision_entropy([500, -300, -400]) < 0.2
    # Two equal moves: maxed out at 1.
    assert decision_entropy([50, 50]) == pytest.approx(1.0)


def test_complexity_features_only_move():
    # A single candidate is by definition the only move.
    feats = complexity_features([300], legal_move_count=1)
    assert feats["is_only_move"] is True
    assert feats["eval_gap_1_2"] is None


def test_complexity_features_sharp_choice():
    # Best and second are 5 cp apart; both reasonable, not an only move.
    feats = complexity_features([120, 115, -200], legal_move_count=30, rating=1500)
    assert feats["eval_gap_1_2"] == 5
    assert feats["n_reasonable"] == 2
    assert feats["is_only_move"] is False


# --- Tier B engine round trip --------------------------------------------

def _engine_or_skip():
    import shutil

    import chess.engine

    path = load_config()["stockfish_path"]
    if not shutil.which(path) and not Path(path).exists():
        pytest.skip("Stockfish binary not installed")
    return chess.engine.SimpleEngine.popen_uci(path)


@pytest.mark.engine
def test_only_move_position_detected():
    from chess_strength.complexity import FenAnalyzer

    cfg = load_config()
    # White king on h1, Black queen g2 giving check, only legal move Kxg2.
    fen = "7k/8/8/8/8/8/6q1/7K w - - 0 1"
    engine = _engine_or_skip()
    with engine:
        analyzer = FenAnalyzer(engine, nodes=cfg["stockfish_nodes"], multipv=cfg["multipv"])
        cps = analyzer.candidate_cps(fen)
        # Same FEN again is a cache hit, engine not called twice.
        analyzer.candidate_cps(fen)
        assert analyzer.hits == 1

    import chess

    legal = chess.Board(fen).legal_moves.count()
    feats = complexity_features(cps, legal_move_count=legal)
    assert feats["is_only_move"] is True
