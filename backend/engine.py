"""Local Stockfish read of a set of positions.

For each FEN we ask Stockfish for its top few moves at a fixed node budget and
keep the side-to-move centipawn scores. From that one read we get everything the
classifier needs about the position:

- the position's own evaluation (the best move's score), used to fill in a move
  quality when the PGN carried no `[%eval]`,
- the decision entropy over the candidate evals, which is the complexity input
  the think-time mapping was trained on,
- the gap between the best two moves, used by the only-move (Great) rule.

The node budget is fixed on purpose so runs reproduce, and it matches the budget
the think-time mapping was fitted at. Do not change it without refitting that
mapping. Positions are analyzed in a process pool for a long game, one
single-threaded engine per worker, never oversubscribed. A short game or a fully
cached re-run uses no pool at all.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable

import chess
import chess.engine

from chess_strength.complexity import complexity_features, decision_entropy

# Locked budget: matches data/processed/thinktime training (30k nodes, multipv 4).
# ponytail: single fixed budget, both eval and entropy come from it. If computed
# evals read as too coarse at Checkpoint A, add a deeper single-PV quality pass.
ENGINE_NODES = 30000
ENGINE_MULTIPV = 4
MATE_CP = 10000

_BATCH = 40  # positions per work unit, ~1.5s at 30k nodes

# Worker-global engine, opened once per process by the pool initializer.
_ENGINE: chess.engine.SimpleEngine | None = None


def _open(sf_path: str, hash_mb: int) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 1, "Hash": hash_mb})
    return eng


def _init_worker(sf_path: str, hash_mb: int) -> None:
    global _ENGINE
    _ENGINE = _open(sf_path, hash_mb)


def _analyse_one(eng: chess.engine.SimpleEngine, fen: str) -> tuple[str, list[int], int]:
    board = chess.Board(fen)
    infos = eng.analyse(board, chess.engine.Limit(nodes=ENGINE_NODES), multipv=ENGINE_MULTIPV)
    cps = sorted(
        (i["score"].relative.score(mate_score=MATE_CP) for i in infos), reverse=True
    )
    return fen, cps, board.legal_moves.count()


def _analyse_batch(fens: list[str]) -> list[tuple[str, list[int], int]]:
    assert _ENGINE is not None
    return [_analyse_one(_ENGINE, fen) for fen in fens]


def features_from_cps(cps: list[int], n_legal: int, white_to_move: bool) -> dict:
    """The three readings the classifier needs, from one position's candidate evals."""
    best = cps[0]
    return {
        "eval_white": best if white_to_move else -best,
        "entropy": decision_entropy(cps),
        "eval_gap": complexity_features(cps, n_legal)["eval_gap_1_2"] or 0.0,
        "n_legal": n_legal,
    }


def analyze_fens(
    fens: list[str],
    cache,
    sf_path: str,
    workers: int,
    hash_mb: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """Stockfish features for every FEN, reading and writing the cache.

    Cached positions are skipped. Only the misses hit the engine, in a pool
    sized to the batch count so a small game does not spawn idle workers.
    """
    fens = list(dict.fromkeys(fens))
    total = len(fens)
    cps_by_fen: dict[str, tuple[list[int], int]] = {}
    missing: list[str] = []
    for fen in fens:
        hit = cache.get_engine(fen, ENGINE_NODES, ENGINE_MULTIPV)
        if hit is None:
            missing.append(fen)
        else:
            cps_by_fen[fen] = hit

    done = total - len(missing)
    if progress:
        progress(done, total)

    if missing:
        batches = [missing[i : i + _BATCH] for i in range(0, len(missing), _BATCH)]
        n_workers = max(1, min(workers, len(batches)))

        def record(rows):
            nonlocal done
            for fen, cps, n_legal in rows:
                cache.put_engine(fen, ENGINE_NODES, ENGINE_MULTIPV, cps, n_legal)
                cps_by_fen[fen] = (cps, n_legal)
                done += 1
            if progress:
                progress(done, total)

        if n_workers == 1:
            eng = _open(sf_path, hash_mb)
            try:
                for batch in batches:
                    record([_analyse_one(eng, fen) for fen in batch])
            finally:
                eng.quit()
        else:
            pool = mp.Pool(n_workers, initializer=_init_worker, initargs=(sf_path, hash_mb))
            try:
                for rows in pool.imap_unordered(_analyse_batch, batches):
                    record(rows)
            finally:
                pool.terminate()
                pool.join()

    return {
        fen: features_from_cps(cps, n_legal, fen.split()[1] == "w")
        for fen, (cps, n_legal) in cps_by_fen.items()
    }
