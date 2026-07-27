"""Phase 1 smoke tests: the package imports and config loads with sane values."""

from pathlib import Path

import chess_strength
from chess_strength.config import load_config


def test_package_imports():
    assert chess_strength.__version__


def test_config_loads_expected_keys():
    cfg = load_config()
    assert cfg["regimes"] == ["blitz", "rapid"]
    assert cfg["stockfish_nodes"] == 200000
    assert cfg["multipv"] == 4
    # The win% constant must not be rounded; it is Lichess's exact value.
    assert cfg["winprob_k"] == 0.00368208


def test_config_paths_are_absolute():
    cfg = load_config()
    for p in cfg["paths"].values():
        assert Path(p).is_absolute()


def test_fixture_pgn_has_clk_and_eval():
    text = (Path(__file__).parent / "fixtures" / "mini.pgn").read_text(encoding="utf-8")
    assert "[%clk" in text
    assert "[%eval" in text
    assert "TimeControl" in text
