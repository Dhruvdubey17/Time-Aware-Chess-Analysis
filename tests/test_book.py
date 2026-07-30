"""Phase A opening-book detection against the bundled offline book.

The three BUILD2 scenarios, all deterministic and offline (no network, no
skips): a normal opening stays Book to the right depth then stops, an offbeat
move leaves book, and an early non-theory move is NOT hidden behind a Book label
(the old bug). Plus honest handling when the book file is missing.

The asserted moves were checked against the shipped book: the mainlines are
theory, and Ke7 / Ba6 are not in any named opening, so they leave book.
"""

from __future__ import annotations

import chess
import pytest

from backend import book
from chess_strength.classify import ClassifyConfig, baseline_label


@pytest.fixture(scope="module")
def reader():
    r = book.open_book({})  # no book_path -> the bundled book
    assert r is not None, "bundled opening book should load"
    yield r
    r.close()


def _walk(sans: list[str]) -> tuple[list[str], list[str]]:
    """FENs-before and UCIs for a line given as SANs."""
    board = chess.Board()
    fens, ucis = [], []
    for san in sans:
        mv = board.parse_san(san)
        fens.append(board.fen())
        ucis.append(mv.uci())
        board.push(mv)
    return fens, ucis


def _lookup(reader):
    return lambda fen: book.book_moves(reader, fen)


def test_start_position_has_the_main_first_moves(reader):
    moves = book.book_moves(reader, chess.Board().fen())
    assert {"e2e4", "d2d4", "c2c4", "g1f3"} <= moves


def test_normal_opening_is_book_to_the_right_depth(reader):
    # Giuoco Piano, then a nonsense king move (4.Ke2) that is not theory.
    fens, ucis = _walk(["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "Ke2"])
    in_book, lookups, ok = book.detect_book(fens, ucis, _lookup(reader))
    assert in_book == [True, True, True, True, True, True, False]
    assert lookups == 7 and ok


def test_offbeat_move_leaves_book(reader):
    # 1.e4 e5 2.Nf3 Nc6 3.Ba6, a bishop move off the board's theory.
    fens, ucis = _walk(["e4", "e5", "Nf3", "Nc6", "Ba6"])
    in_book, _, ok = book.detect_book(fens, ucis, _lookup(reader))
    assert in_book == [True, True, True, True, False] and ok


def test_early_non_theory_move_is_not_labeled_book(reader):
    # The heart of the old bug: 1.e4 e5 2.Nf3 Ke7, a weak early king move. The
    # old rule stamped early moves Book and hid them; now it is not book, so it
    # gets its true label.
    fens, ucis = _walk(["e4", "e5", "Nf3", "Ke7"])
    in_book, _, ok = book.detect_book(fens, ucis, _lookup(reader))
    assert in_book == [True, True, True, False] and ok

    c = ClassifyConfig()
    # A big-loss non-book move keeps its real tier instead of being called Book.
    assert baseline_label(35.0, in_book[3], False, False, c) == "Blunder"
    # If it were flagged book it would read Book. That masking is what this fixes.
    assert baseline_label(35.0, True, False, False, c) == "Book"


def test_missing_book_does_not_guess():
    # If the book cannot be loaded, we report it and do NOT fall back to ply count.
    assert book.open_book({"book_path": "/no/such/book.bin"}) is None
    fens, ucis = _walk(["e4", "e5", "Nf3"])
    in_book, lookups, ok = book.detect_book(fens, ucis, lambda fen: None)
    assert in_book == [False, False, False]
    assert lookups == 0 and ok is False
