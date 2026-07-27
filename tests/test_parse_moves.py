"""Phase 3 tests: per-move parsing and point-of-view eval normalization.

Runs offline against the plain-PGN fixture. The three usable fixture games
parse into a known number of ply rows, and the POV flip is checked directly.
"""

from pathlib import Path

import chess
import chess.pgn

from chess_strength.parse_moves import parse_game, parse_pgn_file, to_pov_cp

FIXTURE = Path(__file__).parent / "fixtures" / "mini.pgn"


def test_to_pov_cp_flips_on_black_move():
    # White-POV +2.0 (200 cp) on Black's move reads as -200 for Black.
    assert to_pov_cp(200, mover_is_white=False) == -200
    assert to_pov_cp(200, mover_is_white=True) == 200
    assert to_pov_cp(-50, mover_is_white=False) == 50


def _rows():
    return list(parse_pgn_file(FIXTURE))


def test_only_usable_games_parsed():
    rows = _rows()
    # The eval-less fourth game (grace/heidi) must not appear.
    assert {r["game_id"] for r in rows} == {"aaaa1111", "bbbb2222", "cccc3333"}


def test_ply_counts_per_game():
    rows = _rows()
    by_game: dict[str, int] = {}
    for r in rows:
        by_game[r["game_id"]] = by_game.get(r["game_id"], 0) + 1
    # 16, 14, 12 half-moves in the three fixture games.
    assert by_game == {"aaaa1111": 16, "bbbb2222": 14, "cccc3333": 12}


def test_no_null_eval_or_clock():
    for r in _rows():
        assert r["eval_cp"] is not None
        assert r["clock_s"] is not None


def test_pov_matches_color_on_a_real_row():
    rows = _rows()
    # Game bbbb2222 ply 12 is Rxf7 by Black at White-POV -2.05 (-205 cp),
    # which is +205 from Black's point of view.
    row = next(r for r in rows if r["game_id"] == "bbbb2222" and r["ply"] == 12)
    assert row["color"] == "black"
    assert row["eval_cp_white"] == -205
    assert row["eval_cp"] == 205


def test_first_ply_fields():
    rows = _rows()
    first = next(r for r in rows if r["game_id"] == "aaaa1111" and r["ply"] == 1)
    assert first["color"] == "white"
    assert first["san"] == "e4"
    assert first["uci"] == "e2e4"
    assert first["move_number"] == 1
    assert first["fen_before"] == chess.STARTING_FEN
    assert first["clock_s"] == 180.0
