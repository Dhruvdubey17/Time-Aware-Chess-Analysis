"""Build the bundled opening book from the lichess-org/chess-openings data.

The source is the Lichess opening classification (CC0, public domain): five TSV
files of named ECO openings, each a short line of moves. We replay every line
and record each (position, move) pair as a Polyglot entry. That gives a compact
membership book: "is this move theory from this position", which is all the book
detector needs. We do NOT need master-game frequency stats, so every entry gets
weight 1.

    python scripts/build_opening_book.py

Writes assets/book/openings.bin. Re-run to rebuild if the source updates. The
TSVs are cached under data/raw/openings so a rebuild does not re-download.
"""

from __future__ import annotations

import io
import struct
import urllib.request
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "book" / "openings.bin"
CACHE = ROOT / "data" / "raw" / "openings"
BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
FILES = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]

_PROMO = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


def polyglot_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a move in Polyglot's 16-bit format. Castling is stored as the king
    moving onto its own rook, which is the Polyglot convention python-chess reads
    back."""
    to_sq = move.to_square
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        to_sq = chess.square(7 if kingside else 0, rank)
    promo = _PROMO.get(move.promotion, 0)
    return (chess.square_file(to_sq)
            | (chess.square_rank(to_sq) << 3)
            | (chess.square_file(move.from_square) << 6)
            | (chess.square_rank(move.from_square) << 9)
            | (promo << 12))


def load_tsv(name: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / name
    if not local.exists():
        req = urllib.request.Request(f"{BASE}/{name}",
                                     headers={"User-Agent": "opening-book-build/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            local.write_bytes(r.read())
    return local.read_text(encoding="utf-8")


def collect_entries() -> dict[tuple[int, int], int]:
    """(zobrist key, encoded move) -> weight, deduped across all opening lines."""
    entries: dict[tuple[int, int], int] = {}
    lines = 0
    for name in FILES:
        text = load_tsv(name)
        for row in text.splitlines()[1:]:  # skip the header
            parts = row.split("\t")
            if len(parts) < 3:
                continue
            game = chess.pgn.read_game(io.StringIO(parts[2]))
            if game is None:
                continue
            lines += 1
            board = game.board()
            for move in game.mainline_moves():
                key = chess.polyglot.zobrist_hash(board)
                entries[(key, polyglot_move(board, move))] = 1
                board.push(move)
    print(f"read {lines} opening lines, {len(entries)} unique position-move entries")
    return entries


def write_book(entries: dict[tuple[int, int], int]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Polyglot readers binary-search by key, so entries must be sorted by key.
    rows = sorted(entries.items())
    with OUT.open("wb") as f:
        for (key, move), weight in rows:
            f.write(struct.pack(">QHHI", key, move, weight, 0))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


def verify() -> None:
    """Read the book back and confirm a mainline and a castling line are found,
    so the encoding (including Polyglot's king-to-rook castling) is correct."""
    with chess.polyglot.open_reader(OUT) as reader:
        start = chess.Board()
        first = {e.move.uci() for e in reader.find_all(start)}
        assert "e2e4" in first and "d2d4" in first, first

        # Ruy Lopez, castling into book: 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O
        b = chess.Board()
        for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]:
            b.push_san(san)
        castle = b.parse_san("O-O")
        book_here = {e.move.uci() for e in reader.find_all(b)}
        assert castle.uci() in book_here, (castle.uci(), book_here)
    print("verify ok: mainline first moves and a castling book move both read back")


if __name__ == "__main__":
    write_book(collect_entries())
    verify()
