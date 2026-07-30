"""Phase 1 intake and detection, one test per branch the PGN can take."""

from pathlib import Path

from backend.intake import MATE_CP, parse_pgn, parse_pgn_file

FIXTURES = Path(__file__).parent / "fixtures"


def _one(name: str):
    reports = parse_pgn_file(FIXTURES / name)
    assert len(reports) == 1
    return reports[0]


# --- Lichess with clocks and evals: the full-capability path ---------------

def test_lichess_clk_eval_is_fully_capable():
    r = _one("lichess_clk_eval.pgn")
    assert r.accepted and r.reject_reason is None
    assert r.site == "lichess"
    assert r.regime == "blitz"
    assert r.has_clocks and r.has_evals
    assert r.time_aware_available
    assert "time-aware review is available" in r.capability_note
    assert r.n_plies == 23 and r.n_moves == 12


def test_lichess_normalized_move_fields():
    r = _one("lichess_clk_eval.pgn")
    first = r.moves[0]
    assert first.ply == 1 and first.move_number == 1
    assert first.side == "white" and first.san == "e4"
    assert first.uci == "e2e4"
    assert first.fen_before.startswith("rnbqkbnr/pppppppp")
    assert first.clock_s == 180.0  # 0:03:00
    assert first.eval_cp_white == 15  # 0.15 pawns -> 15 cp, White POV
    assert first.phase == "opening"


def test_intake_leaves_book_to_analysis():
    # The old rule stamped the first N plies as Book from the ply number alone,
    # which hid real early blunders. Parsing no longer decides book at all; real
    # theory detection is a live per-position lookup in the analysis step.
    from dataclasses import fields

    r = _one("lichess_clk_eval.pgn")
    assert "in_book" not in {f.name for f in fields(r.moves[0])}
    assert r.moves[0].phase == "opening"  # phase still works as before


def test_eval_stored_white_pov_for_black_move():
    # Black's 11th move (ply 22) is ...Bxe5 with eval 1.85 -> +185 cp White POV,
    # even though Black is the mover. The sign is White's, like the raw [%eval].
    r = _one("lichess_clk_eval.pgn")
    black_move = next(m for m in r.moves if m.ply == 22)
    assert black_move.side == "black"
    assert black_move.eval_cp_white == 185


# --- chess.com with clocks only: evals must be computed later --------------

def test_chesscom_clk_only():
    r = _one("chesscom_clk.pgn")
    assert r.accepted
    assert r.site == "chess.com"
    assert r.regime == "blitz"  # TimeControl "300"
    assert r.has_clocks and not r.has_evals
    assert r.time_aware_available
    assert "computed on your machine" in r.capability_note
    assert r.moves[0].clock_s == 300.0
    assert r.moves[0].eval_cp_white is None
    assert r.result == "0-1"
    assert "resignation" in r.termination
    assert r.white_elo == 1123 and r.black_elo == 1150


# --- bare PGN: no clocks, no evals -----------------------------------------

def test_bare_pgn_baseline_only():
    r = _one("bare.pgn")
    assert r.accepted
    assert r.site == "unknown"
    assert r.regime is None
    assert not r.has_clocks and not r.has_evals
    assert not r.time_aware_available
    assert "no move clocks" in r.capability_note
    assert "computed on your machine" in r.capability_note
    assert r.white_elo is None
    assert r.n_plies == 40 and r.n_moves == 20
    assert r.moves[0].clock_s is None and r.moves[0].eval_cp_white is None


# --- incomplete / abandoned game: still parses honestly --------------------

def test_incomplete_game_parses_what_it_has():
    r = _one("incomplete.pgn")
    assert r.accepted
    assert r.result == "*"
    assert r.termination == "Abandoned"
    assert r.regime == "rapid"  # 600+5
    assert r.n_plies == 5 and r.n_moves == 3
    assert r.time_aware_available  # clocks present, rapid


# --- variants and non-standard boards: rejected with a clear reason --------

def test_variants_are_rejected():
    reports = parse_pgn_file(FIXTURES / "variant.pgn")
    assert len(reports) == 2
    chess960, from_position = reports
    assert not chess960.accepted
    assert "Chess960" in chess960.reject_reason
    assert chess960.moves == []
    assert not from_position.accepted
    assert "normal chess position" in from_position.reject_reason


# --- several games in one paste: the picker path ---------------------------

def test_multi_game_file_returns_each_game():
    reports = parse_pgn_file(FIXTURES / "mini.pgn")
    assert len(reports) == 4
    assert all(r.accepted for r in reports)
    assert [r.index for r in reports] == [0, 1, 2, 3]
    # The 4th game has clocks but no evals (a clk-only Lichess game).
    assert reports[3].has_clocks and not reports[3].has_evals
    assert reports[3].result == "*"


# --- odds and ends ---------------------------------------------------------

def test_empty_and_blank_input_give_no_games():
    assert parse_pgn("") == []
    assert parse_pgn("   \n\t  \n") == []


def test_mate_eval_maps_to_the_extreme():
    pgn = '[Event "t"]\n[Site "?"]\n\n1. e4 { [%eval #3] } e5 *'
    r = parse_pgn(pgn)[0]
    assert r.moves[0].eval_cp_white is not None
    assert r.moves[0].eval_cp_white > MATE_CP - 100  # near +10000, a White mate


def test_summary_drops_the_move_payload():
    r = _one("lichess_clk_eval.pgn")
    s = r.summary()
    assert "moves" not in s
    assert s["site"] == "lichess" and s["time_aware_available"]


# --- edge cases -----------------------------------------------------------

def test_classical_clocks_but_out_of_scope():
    # Clocks present, but classical is outside the blitz/rapid calibration, so
    # time-aware is off with a plain reason (never a faked pressure signal).
    r = _one("classical.pgn")
    assert r.accepted and r.has_clocks
    assert r.regime == "classical"
    assert not r.time_aware_available
    assert "blitz and rapid" in r.capability_note


def test_malformed_pgn_keeps_what_parsed():
    r = _one("malformed.pgn")
    assert r.accepted
    assert r.n_plies == 5  # e4 e5 Nf3 Nc6 Bb5, then the junk token is dropped
    assert "could not be read" in r.capability_note


def test_daily_time_control_falls_back_to_baseline():
    pgn = ('[Event "Correspondence"]\n[Site "?"]\n[Result "1-0"]\n'
           '[TimeControl "1/259200"]\n\n1. d4 d5 2. c4 e6 1-0\n')
    r = parse_pgn(pgn)[0]
    assert r.accepted and r.regime is None
    assert not r.time_aware_available
    assert "no move clocks" in r.capability_note


def test_zero_move_game_is_honest():
    r = parse_pgn('[Event "empty"]\n[Site "?"]\n[Result "*"]\n\n*\n')[0]
    assert r.accepted and r.n_plies == 0
    assert "no moves" in r.capability_note


def test_random_text_yields_no_usable_game():
    reports = parse_pgn("this is not a chess game at all, just words")
    assert reports == [] or all(r.n_plies == 0 for r in reports)
