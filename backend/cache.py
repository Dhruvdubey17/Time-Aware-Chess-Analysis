"""A small on-disk cache keyed by FEN, so a repeated position is never analyzed
twice and re-reviewing a game is instant.

Two kinds of result are cached: the Stockfish read of a position (its top-move
evals and legal-move count) and the Maia human-move dispersion for a position at
a rating band. SQLite is plenty here: one local user, small tables, and it
survives restarts with no server to run.

The cache is used from the main analysis thread only. The engine work happens in
a separate process pool that just computes and hands results back; the main
thread does all the reads and writes, so one connection per analysis is safe.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class FenCache:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS engine "
            "(fen TEXT, nodes INT, multipv INT, cps TEXT, n_legal INT, "
            "PRIMARY KEY (fen, nodes, multipv))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS maia "
            "(fen TEXT, regime TEXT, band_elo INT, entropy REAL, "
            "PRIMARY KEY (fen, regime, band_elo))"
        )
        self._conn.commit()

    def get_engine(self, fen: str, nodes: int, multipv: int) -> tuple[list[int], int] | None:
        row = self._conn.execute(
            "SELECT cps, n_legal FROM engine WHERE fen=? AND nodes=? AND multipv=?",
            (fen, nodes, multipv),
        ).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def put_engine(self, fen: str, nodes: int, multipv: int, cps: list[int], n_legal: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO engine VALUES (?, ?, ?, ?, ?)",
            (fen, nodes, multipv, json.dumps(cps), n_legal),
        )
        self._conn.commit()

    def get_maia(self, fen: str, regime: str, band_elo: int) -> float | None:
        row = self._conn.execute(
            "SELECT entropy FROM maia WHERE fen=? AND regime=? AND band_elo=?",
            (fen, regime, band_elo),
        ).fetchone()
        return row[0] if row else None

    def put_maia(self, fen: str, regime: str, band_elo: int, entropy: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO maia VALUES (?, ?, ?, ?)",
            (fen, regime, band_elo, entropy),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
