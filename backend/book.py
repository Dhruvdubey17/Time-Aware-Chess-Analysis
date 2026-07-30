"""Opening-book detection from a small bundled offline book.

A move is Book only when the bundled opening book has an entry for the position
whose move matches the move played. This works at any move number and stops
calling a move Book the moment the game leaves theory. It replaces the old
"first N plies are Book" rule, which stamped Book on non-theory moves and, worse,
hid real early blunders behind a Book label.

The book is a Polyglot .bin read with python-chess. It is local, so there is no
network, no rate limit, and no outage to handle. We only need membership (is this
move theory here), not master-game frequency, so a compact book is enough. If the
book file is missing or unreadable we do not guess: those moves get their normal
eval-based label and the report says the book could not be loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import chess
import chess.polyglot

# The book that ships with the app. config may override with book_path.
_BUNDLED = Path(__file__).resolve().parents[1] / "assets" / "book" / "openings.bin"


def book_path(cfg: dict) -> Path:
    return Path(cfg.get("book_path") or _BUNDLED)


def open_book(cfg: dict):
    """Open the bundled book for reading, or None if it is missing or unreadable.
    None means "book unavailable", which the caller reports honestly rather than
    guessing at theory."""
    path = book_path(cfg)
    try:
        return chess.polyglot.open_reader(path)
    except (OSError, ValueError):
        return None


def book_moves(reader, fen: str) -> set[str]:
    """UCI moves the book lists for this position, empty when the position is not
    in the book. python-chess decodes Polyglot castling back to the normal
    king-to-castle-square form, so these UCIs match a played move's uci()."""
    return {entry.move.uci() for entry in reader.find_all(chess.Board(fen))}


def detect_book(fens: list[str], ucis: list[str],
                lookup: Callable[[str], set[str] | None]) -> tuple[list[bool], int, bool]:
    """Walk the game in order and mark each move book or not.

    A move is book when its position is in the book and the played move appears
    there. We stop at the first move that leaves theory. A line that leaves book
    and transposes back into it is rare, and not worth a lookup on every later
    position to catch.

    Returns (in_book per move, positions looked up, book_ok). book_ok is False
    only when the book was unavailable, so the caller knows those moves were
    labeled without a theory check.
    """
    n = len(fens)
    in_book = [False] * n
    lookups = 0
    for i in range(n):
        moves = lookup(fens[i])
        if moves is None:
            return in_book, lookups, False  # book unavailable; rest unchecked
        lookups += 1
        if ucis[i] in moves:
            in_book[i] = True
        else:
            break  # left theory, so every later move is out of book too
    return in_book, lookups, True
