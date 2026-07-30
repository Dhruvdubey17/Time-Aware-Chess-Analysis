"""Phase E premove and sub-floor detection.

The BUILD2 scenarios: a chess.com premove flagged confidently (tenths clocks), a
Lichess sub-second move flagged ambiguous (whole-second clocks), an increment
game where the clock rises, and the hard guardrail that a premove or ambiguous
move can never be upgraded. Pure clock arithmetic, so all deterministic.
"""

from __future__ import annotations

from backend import premove


def test_detects_subsecond_precision():
    assert premove.has_subsecond_clocks([180.0, 179.0, 60.0]) is False  # Lichess
    assert premove.has_subsecond_clocks([59.9, 58.3]) is True  # chess.com tenths
    assert premove.has_subsecond_clocks([None, 60.0, 59.9]) is True  # ignores None


def test_chesscom_premove_flagged_confidently():
    # Tenths clocks: a move costing about 0.1s is a premove, a real think is not.
    assert premove.classify(0.1, is_first_move=False, subsecond=True) == premove.PREMOVE
    assert premove.classify(0.0, is_first_move=False, subsecond=True) == premove.PREMOVE
    assert premove.classify(2.5, is_first_move=False, subsecond=True) == premove.GENUINE


def test_lichess_subsecond_is_ambiguous_not_genuine():
    # Whole-second clocks: under a second rounds to 0 and cannot be told from a
    # premove, so it is ambiguous, never a confident genuine find.
    assert premove.classify(0.0, is_first_move=False, subsecond=False) == premove.SUB_FLOOR_AMBIGUOUS
    assert premove.classify(1.0, is_first_move=False, subsecond=False) == premove.GENUINE
    assert premove.classify(5.0, is_first_move=False, subsecond=False) == premove.GENUINE


def test_increment_clock_rise_is_a_premove_signature():
    # 2+1: a premove spends ~0.1s but is credited 1s, so the clock rises. With
    # tenths, move_time = clock_before + inc - clock_after = 30.0 + 1 - 30.9 = 0.1.
    assert premove.classify(30.0 + 1 - 30.9, is_first_move=False, subsecond=True) == premove.PREMOVE
    # Same rise on whole-second Lichess clocks (30 + 1 - 31 = 0) is only ambiguous.
    assert premove.classify(30 + 1 - 31, is_first_move=False, subsecond=False) == premove.SUB_FLOOR_AMBIGUOUS


def test_first_move_is_not_classified():
    # Increment credit on move one differs by server, and move one has a full
    # clock so it can never be upgraded; a tiny move time there is not a premove.
    assert premove.classify(0.0, is_first_move=True, subsecond=True) == premove.GENUINE
    assert premove.classify(0.05, is_first_move=True, subsecond=True) == premove.GENUINE


def test_missing_clock_is_unknown():
    assert premove.classify(None, is_first_move=False, subsecond=True) == premove.UNKNOWN


def test_only_genuine_can_be_upgraded():
    # The hard guardrail: nothing but a genuine at-the-board move may upgrade.
    assert premove.can_be_under_pressure(premove.GENUINE) is True
    assert premove.can_be_under_pressure(premove.PREMOVE) is False
    assert premove.can_be_under_pressure(premove.SUB_FLOOR_AMBIGUOUS) is False
    assert premove.can_be_under_pressure(premove.UNKNOWN) is False
