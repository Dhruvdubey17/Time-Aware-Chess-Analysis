"""Phase 2 analysis service.

Fast unit tests for the pure assembly bits, then end-to-end runs on the Phase 1
fixtures. The end-to-end tests need the real Stockfish binary (marker `engine`)
and, for the time-aware path, the Maia environment (marker `maia`); both skip
cleanly when the machine does not have them.
"""

from pathlib import Path

import pytest

from backend import maia_client
from backend.analyze import (
    _parse_tc,
    _sac_series,
    analyze,
    band_label,
    difficulty_bucket,
)
from backend.cache import FenCache
from backend.intake import parse_pgn_file
from chess_strength.classify import ClassifyConfig
from chess_strength.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE_LABELS = {"Book", "Best", "Excellent", "Good", "Inaccuracy", "Mistake",
                   "Blunder", "Great", "Brilliant"}


# --- fast unit tests -------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    cache = FenCache(tmp_path / "c.sqlite")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert cache.get_engine(fen, 30000, 4) is None
    cache.put_engine(fen, 30000, 4, [48, 31, 26], 20)
    assert cache.get_engine(fen, 30000, 4) == ([48, 31, 26], 20)
    assert cache.get_maia(fen, "blitz", 1600) is None
    cache.put_maia(fen, "blitz", 1600, 1.23)
    assert cache.get_maia(fen, "blitz", 1600) == 1.23
    cache.close()


def test_band_label():
    assert band_label(1200) == "[0, 1300)"
    assert band_label(1600) == "[1500, 1700)"
    assert band_label(3000) == "[2100, 9999)"


def test_parse_tc():
    assert _parse_tc("180+2") == (180, 2)
    assert _parse_tc("300") == (300, 0)  # chess.com style, no increment
    assert _parse_tc("") == (0, 0)
    assert _parse_tc("1/259200") == (0, 0)  # daily, unparseable base


def test_difficulty_bucket():
    c = ClassifyConfig()
    assert difficulty_bucket(None, c) is None
    assert difficulty_bucket(0.5, c) == "routine"
    assert difficulty_bucket(2.3, c) == "very hard"


def test_sac_series_ignores_recovered_material():
    # A sacrifice counts only if the material is not won back within the mover's
    # next two turns (best recovery over that window).
    class M:
        def __init__(self, side):
            self.side = side
    moves = [M("white"), M("black"), M("white"), M("black"), M("white"), M("black")]
    # White drops 320 but wins it back by its next-but-one turn -> not a sacrifice.
    recovered = [320, 0, 0, 0, 320, 0]  # white turns: 320 -> 0 -> 320
    assert _sac_series(moves, recovered)[0] == 0.0
    # White drops 320 and stays down -> a real sacrifice.
    sustained = [320, 0, 0, 0, 0, 0]  # white turns: 320 -> 0 -> 0
    assert _sac_series(moves, sustained)[0] == 320.0


# --- end-to-end on the Phase 1 fixtures ------------------------------------

def _skip_if_no_engine():
    import shutil

    path = load_config()["stockfish_path"]
    if not shutil.which(path) and not Path(path).exists():
        pytest.skip("Stockfish binary not installed")


def _skip_if_no_maia():
    if not maia_client.available():
        pytest.skip("Maia environment (.venv_maia) not present")


def _analyze(name, **kw):
    report = parse_pgn_file(FIXTURES / name)[0]
    return analyze(report, load_config(), **kw)


def _valid_labels(result):
    for m in result.moves:
        assert m["baseline_label"] in BASELINE_LABELS
        assert m["wpl"] >= 0.0
        assert isinstance(m["eval_white"], int)  # always filled, PGN eval or engine


@pytest.mark.engine
def test_baseline_only_when_no_clocks():
    _skip_if_no_engine()
    r = _analyze("bare.pgn")  # no clocks, no evals
    assert not r.time_aware_available
    assert "no move clocks" in r.time_aware_note
    _valid_labels(r)
    # No clocks means no pressure upgrade on any move, whatever else we compute.
    assert all(m["time_aware_label"] is None for m in r.moves)
    assert r.summary["n_upgrades"] == 0


@pytest.mark.engine
@pytest.mark.maia
def test_computes_missing_evals_and_runs_time_aware():
    _skip_if_no_engine()
    _skip_if_no_maia()
    # Clocks present, evals must be computed, and the game leaves book so there
    # are real non-book moves for the time-aware layer to score.
    r = _analyze("chesscom_midgame.pgn")
    assert r.time_aware_available
    _valid_labels(r)
    # Every non-book move got a difficulty score and a time-aware label.
    scored = [m for m in r.moves if not m["in_book"]]
    assert scored and all(m["maia_entropy"] is not None for m in scored)
    assert all(m["time_aware_label"] is not None for m in r.moves)
    assert all(m["residual_s"] is not None for m in scored)


@pytest.mark.engine
@pytest.mark.maia
def test_uses_pgn_evals_and_respects_upgrade_invariants():
    _skip_if_no_engine()
    _skip_if_no_maia()
    r = _analyze("lichess_clk_eval.pgn")
    assert r.time_aware_available
    _valid_labels(r)
    # Any upgrade must be a real one: strictly better label on a low clock, and
    # never manufactured on a full clock. This is the locked calibration's guard.
    for m in r.moves:
        if m["upgraded"]:
            assert m["clock_before_s"] is not None and m["clock_before_s"] < 60.0
            assert m["pressure"] and m["pressure"] > 0.0
            assert m["wpl"] <= 10.0  # only good moves are eligible
            assert m["premove_status"] == "genuine"  # premoves can never upgrade


@pytest.mark.engine
@pytest.mark.maia
def test_bullet_chesscom_time_aware_and_premoves_never_upgraded():
    # The real chess.com bullet game (tenths), so premoves are detected and the
    # time-aware review runs. The one thing that must never happen: a premove
    # earning a found-under-pressure upgrade.
    _skip_if_no_engine()
    _skip_if_no_maia()
    r = _analyze("chesscom_bullet.pgn")
    assert r.regime == "bullet" and r.time_aware_available
    assert "reliable" in r.time_aware_note  # tenths clocks
    _valid_labels(r)
    premoves = [m for m in r.moves if m["premove_status"] == "premove"]
    assert premoves  # this game has real premoves
    assert all(not m["upgraded"] for m in premoves)
    assert all(m["under_pressure"] in (None, "insufficient_evidence") for m in premoves)
    # Every upgrade is a genuine, at-the-board move, never a slip.
    for m in r.moves:
        if m["upgraded"]:
            assert m["premove_status"] == "genuine" and not m["misclick_suspect"]
            assert m["under_pressure"] == "found_under_pressure"


@pytest.mark.engine
@pytest.mark.maia
def test_bullet_lichess_whole_second_is_limited():
    # Public Lichess bullet is whole-second, so sub-second moves are ambiguous and
    # the review says so plainly; none of them may be upgraded.
    _skip_if_no_engine()
    _skip_if_no_maia()
    r = _analyze("lichess_bullet.pgn")
    assert r.regime == "bullet" and r.time_aware_available
    assert "limited" in r.time_aware_note  # whole-second honesty
    _valid_labels(r)
    for m in r.moves:
        if m["premove_status"] in ("premove", "sub_floor_ambiguous"):
            assert not m["upgraded"]


@pytest.mark.engine
def test_short_game_ending_in_checkmate():
    # Exercises the terminal-checkmate eval path (no [%eval], so the mate must be
    # scored directly) and a very short game, without crashing.
    _skip_if_no_engine()
    r = _analyze("short_mate.pgn")
    _valid_labels(r)
    assert not r.time_aware_available  # no clocks
    last = r.moves[-1]
    assert last["san"] == "Qxf7#"
    assert last["win_after"] == 100.0  # the mate reads as a full win for the mover
    assert last["eval_white"] >= 9000  # mate mapped to the extreme, White POV


@pytest.mark.engine
def test_cache_makes_reanalysis_hit_the_cache(tmp_path):
    # Second run of the same game should find every engine FEN already cached.
    _skip_if_no_engine()
    cfg = load_config()
    report = parse_pgn_file(FIXTURES / "bare.pgn")[0]
    analyze(report, cfg)  # warms the cache
    from backend import engine as eng
    from backend.cache import FenCache
    fens = [m.fen_before for m in report.moves]
    cache = FenCache(Path(cfg["paths"]["processed"]) / "app_cache.sqlite")
    try:
        assert all(cache.get_engine(f, eng.ENGINE_NODES, eng.ENGINE_MULTIPV) is not None
                   for f in fens)
    finally:
        cache.close()
