"""Phase 4 tests: per-move time spent.

Values are checked against the fixture by hand, including the increment game
where the clock rises across moves and the pre-move clamp at zero.
"""

from pathlib import Path

import pytest

from chess_strength.parse_moves import parse_pgn_file
from chess_strength.timespent import (
    add_time_spent,
    parse_time_control,
    time_spent_series,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini.pgn"


def test_parse_time_control():
    assert parse_time_control("180+2") == (180, 2)
    assert parse_time_control("300+0") == (300, 0)
    assert parse_time_control("600+5") == (600, 5)


def test_parse_time_control_rejects_correspondence():
    with pytest.raises(ValueError):
        parse_time_control("-")


def test_series_first_move_uses_base():
    # 180+2: clocks stay flat-ish because the 2s increment offsets thinking.
    clocks = [180, 179, 178, 178, 177, 177, 176, 176]
    assert time_spent_series(clocks, 180, 2) == [2, 3, 3, 2, 3, 2, 3, 2]


def test_series_handles_increment_rise():
    # 600+5: the clock climbs from 600 to 603 on move 2, yet time spent stays a
    # sane positive 2s, not a negative.
    clocks = [600, 603, 598, 601, 604, 606]
    assert time_spent_series(clocks, 600, 5) == [5, 2, 10, 2, 2, 3]


def test_series_clamps_rounding_negatives():
    # Clock cannot really rise more than the increment; a rounding blip clamps.
    assert time_spent_series([102], 100, 0) == [0.0]


def test_add_time_spent_on_fixture():
    rows = add_time_spent(list(parse_pgn_file(FIXTURE)))
    white_g1 = [r["time_spent_s"] for r in rows
                if r["game_id"] == "aaaa1111" and r["color"] == "white"]
    assert white_g1 == [2, 3, 3, 2, 3, 2, 3, 2]

    # The increment-rise move in the rapid game.
    white_g3 = [r["time_spent_s"] for r in rows
                if r["game_id"] == "cccc3333" and r["color"] == "white"]
    assert white_g3 == [5, 2, 10, 2, 2, 3]
