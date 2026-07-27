"""Sanity-check the Stockfish binary.

Opens the engine named in the config and prints a bestmove for the start
position. If this prints a move, the engine is wired up correctly and later
phases can call it.
"""

from __future__ import annotations

import sys

import chess
import chess.engine

from chess_strength.config import load_config


def main() -> int:
    cfg = load_config()
    engine_path = cfg["stockfish_path"]
    nodes = cfg["stockfish_nodes"]

    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except FileNotFoundError:
        print(f"Stockfish not found at {engine_path!r}. "
              "Install it and set stockfish_path in config/default.yaml.")
        return 1

    # Fixed node budget, not time, so this is reproducible.
    with engine:
        board = chess.Board()
        result = engine.play(board, chess.engine.Limit(nodes=nodes))
        print(f"bestmove: {result.move.uci()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
