"""Phase 6 tests: feature assembly, flags, no leakage, filtered view.

Runs offline on the fixture. Checks the join produces the expected columns, that
the player's target Glicko never leaks into the move features, that in_book and
premove_suspect behave, and that the win% before the first move is 50.
"""

from pathlib import Path

import pandas as pd

from chess_strength.config import load_config
from chess_strength.features import (
    assemble_features,
    build_players,
    filter_clean,
    material_cp,
)
from chess_strength.parse_moves import parse_pgn_file

FIXTURE = Path(__file__).parent / "fixtures" / "mini.pgn"


def _features() -> pd.DataFrame:
    cfg = load_config()
    rows = list(parse_pgn_file(FIXTURE))
    return pd.DataFrame(assemble_features(rows, cfg))


def test_material_start_position_is_even():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert material_cp(start, mover_is_white=True) == 0
    # White up a queen reads positive for White, negative for Black.
    white_up_q = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert material_cp(white_up_q, mover_is_white=True) == 900
    assert material_cp(white_up_q, mover_is_white=False) == -900


def test_expected_columns_present():
    df = _features()
    for col in (
        "log_time_spent", "opp_clock_remaining_s", "frac_time_used",
        "win_pct_before", "wpl", "cpl_clipped", "eval_swing", "decisiveness",
        "game_phase", "material", "is_check", "is_capture",
        "player_rating", "opponent_rating", "in_book", "premove_suspect",
    ):
        assert col in df.columns


def test_no_target_leakage_in_move_features():
    df = _features()
    # The held-aside target must never appear as a move feature. Raw Elo columns
    # are dropped too; only the labelled conditioning rating survives.
    assert "glicko" not in df.columns
    assert "white_elo" not in df.columns
    assert "black_elo" not in df.columns
    assert "player_rating" in df.columns


def test_first_move_win_before_is_fifty():
    df = _features()
    first = df[(df["game_id"] == "aaaa1111") & (df["ply"] == 1)].iloc[0]
    assert first["win_pct_before"] == 50.0
    # WPL is never negative anywhere.
    assert (df["wpl"] >= 0).all()


def test_in_book_and_premove_flags():
    df = _features()
    # book_plies is 12; every ply at or below that is in book.
    assert (df[df["ply"] <= 12]["in_book"]).all()
    assert not (df[df["ply"] > 12]["in_book"]).any()
    # premove_suspect fires exactly when time spent is under the threshold. In
    # the fixture that is move 1 of the 0-increment game, where the clock reads
    # full base before and after so the measured time is 0.
    assert (df[df["premove_suspect"]]["time_spent_s"] < 0.3).all()
    assert (df[~df["premove_suspect"]]["time_spent_s"] >= 0.3).all()


def test_filtered_view_drops_book():
    df = _features()
    clean = filter_clean(df)
    assert len(clean) < len(df)
    assert not clean["in_book"].any()


def test_players_table_holds_glicko_aside():
    df = _features()
    players = build_players(df)
    assert "glicko" in players.columns
    assert "mean_wpl" in players.columns
    assert "wpl_dispersion" in players.columns
    # Six distinct players in the three usable fixture games.
    assert len(players) == 6
    # In the fixture each player has a constant rating, so mean equals it.
    alice = players[players["player_id"] == "alice"].iloc[0]
    assert alice["glicko"] == 1650
