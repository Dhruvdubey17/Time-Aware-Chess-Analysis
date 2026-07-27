"""Phase 4 tests: the win-probability primitives.

Checks the sigmoid's shape (equal at 0, monotonic, symmetric, clipped) and that
WPL never credits a good move, matching the Lichess formulas we copied.
"""

from chess_strength.winprob import accuracy, win_pct, wpl


def test_equal_position_is_fifty():
    assert win_pct(0) == 50.0


def test_monotonic_and_symmetric():
    assert win_pct(-100) < win_pct(0) < win_pct(100)
    # win% for a position and its mirror must sum to a full 100.
    for cp in (50, 200, 750):
        assert win_pct(cp) + win_pct(-cp) == 100.0


def test_eval_is_clipped():
    # Past the clip, more centipawns change nothing.
    assert win_pct(5000) == win_pct(1000)
    assert win_pct(-5000) == win_pct(-1000)


def test_wpl_clamps_good_moves_to_zero():
    # A move that raises win% (60 -> 72) lost nothing.
    assert wpl(60.0, 72.0) == 0.0
    # A move that dropped win% by 12 points cost 12.
    assert wpl(60.0, 48.0) == 12.0


def test_accuracy_bounds():
    # A perfect move (no win% drop) is essentially 100% accurate.
    assert accuracy(70.0, 70.0) == 100.0 or abs(accuracy(70.0, 70.0) - 100.0) < 1e-3
    # A total collapse floors at 0, never negative.
    assert accuracy(100.0, 0.0) == 0.0
