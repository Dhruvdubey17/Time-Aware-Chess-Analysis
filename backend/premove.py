"""Premove and sub-floor detection from clock data.

A premove is a move queued before it is the player's turn. The server charges it
a tiny fixed cost (chess.com about 0.1s), so the move was NOT found at the board
under pressure. It must never earn a "found it fast under pressure" upgrade, even
when it is a strong move in a hard position. That false positive is the one thing
bullet support cannot ship with, so this is a hard guardrail.

We can only be confident when the clock has sub-second precision (chess.com
tenths). Public Lichess clocks are whole seconds, so anything under a second
rounds to zero and a true premove cannot be told apart from a genuine sub-second
find. We mark those ambiguous and still refuse the upgrade, because a wrong
"brilliant under pressure" is worse than a missed one.

Pure clock arithmetic, no PGN parsing, so it is easy to test. The move time is
kept signed: in increment formats a premove can make the clock rise, so a move
time at or below zero is itself a premove signature, not a value to clamp away.
"""

from __future__ import annotations

PREMOVE = "premove"
SUB_FLOOR_AMBIGUOUS = "sub_floor_ambiguous"
GENUINE = "genuine"
UNKNOWN = "unknown"

# chess.com charges a premove about 0.1s; allow a little slack for rounding.
_PREMOVE_MAX_S = 0.15


def has_subsecond_clocks(clocks) -> bool:
    """True when any clock reading carries a fraction of a second, so the PGN has
    tenths (chess.com). Whole-second only (public Lichess) returns False."""
    return any(c is not None and abs(c - round(c)) > 1e-6 for c in clocks)


def classify(move_time: float | None, is_first_move: bool, subsecond: bool) -> str:
    """One move's premove status from its signed move time in seconds, where
    move_time = clock_before + increment - clock_after.

    The first move of each side is not classified: increment credit on move one
    differs by server, and a full clock means the move can never be upgraded
    anyway, so guessing there would only pollute the premove rate.
    """
    if move_time is None:
        return UNKNOWN
    if is_first_move:
        return GENUINE
    if subsecond:
        return PREMOVE if move_time <= _PREMOVE_MAX_S else GENUINE
    # Whole-second clocks: under one second is indistinguishable from a premove.
    return SUB_FLOOR_AMBIGUOUS if move_time < 1.0 else GENUINE


def can_be_under_pressure(status: str) -> bool:
    """Only a genuine at-the-board move may earn an under-pressure upgrade."""
    return status == GENUINE
