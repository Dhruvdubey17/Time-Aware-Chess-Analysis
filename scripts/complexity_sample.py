"""Run Tier-B (Stockfish MultiPV) complexity on a sample of positions.

Reads parsed per-move parquet from data/interim/moves, samples up to N unique
FENs, and scores each with Stockfish at the fixed node budget. Prints timing and
cache stats so we can see the run stays bounded. Pass a fixture .pgn instead of
the parquet dir with --pgn for a quick end-to-end check without real data.

Fixed node budget, threads 1, so results are reproducible.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine

from chess_strength.complexity import FenAnalyzer, complexity_features
from chess_strength.config import load_config
from chess_strength.parse_moves import parse_pgn_file


def _fens_from_pgn(path: Path) -> list[dict]:
    return [{"fen": r["fen_before"], "rating": r.get("white_elo")}
            for r in parse_pgn_file(path)]


def _fens_from_parquet(moves_dir: Path) -> list[dict]:
    import pandas as pd

    frames = [pd.read_parquet(p) for p in sorted(moves_dir.glob("*.parquet"))]
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    return [{"fen": r.fen_before, "rating": r.white_elo} for r in df.itertuples()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", type=Path, help="score positions from this PGN instead of parquet")
    ap.add_argument("--n", type=int, default=1000, help="max positions to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()

    if args.pgn:
        positions = _fens_from_pgn(args.pgn)
    else:
        moves_dir = Path(cfg["paths"]["interim"]) / "moves"
        positions = _fens_from_parquet(moves_dir)

    if not positions:
        print("no positions found. Run Phase 2 and 3 on real data first, or pass --pgn.")
        return 1

    random.seed(args.seed)
    random.shuffle(positions)
    positions = positions[: args.n]

    try:
        engine = chess.engine.SimpleEngine.popen_uci(cfg["stockfish_path"])
    except FileNotFoundError:
        print(f"Stockfish not found at {cfg['stockfish_path']!r}.")
        return 1

    engine.configure({"Threads": 1})
    start = time.time()
    only_moves = 0
    with engine:
        analyzer = FenAnalyzer(engine, nodes=cfg["stockfish_nodes"], multipv=cfg["multipv"])
        for pos in positions:
            cps = analyzer.candidate_cps(pos["fen"])
            legal = chess.Board(pos["fen"]).legal_moves.count()
            feats = complexity_features(cps, legal, pos["rating"])
            only_moves += feats["is_only_move"]

    elapsed = time.time() - start
    n = len(positions)
    print(f"scored {n} positions in {elapsed:.1f}s ({elapsed / n:.3f}s each)")
    print(f"cache hits: {analyzer.hits}, unique FENs: {len(analyzer._cache)}")
    print(f"only-move positions: {only_moves}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
