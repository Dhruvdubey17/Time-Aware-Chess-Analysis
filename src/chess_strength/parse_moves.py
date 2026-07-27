"""Phase 3: turn filtered PGN into a tidy per-move table.

We walk each game's mainline and emit one row per ply: who moved, the move
itself, the position before it, and the two annotations we filtered for, the
`[%eval]` and the `[%clk]`. Evals arrive in White's point of view, so we flip
them to the mover's point of view here. That way a later phase can read every
eval the same way regardless of whose turn it was.

Reads the plain `.pgn` shards written in Phase 2 and writes one parquet per
shard under data/interim/moves.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

# Mate is stored as a large signed centipawn value. python-chess offsets it by
# the mate distance, so a mate in 1 scores higher than a mate in 5.
_MATE_CP = 10000


def to_pov_cp(white_cp: int, mover_is_white: bool) -> int:
    """Flip a White-point-of-view centipawn eval to the mover's point of view.

    A +200 (White winning) eval on Black's move reads as -200 for Black.
    """
    return white_cp if mover_is_white else -white_cp


def _game_id(game: chess.pgn.Game, fallback: str) -> str:
    """Lichess game id from the Site URL, e.g. .../abcd1234 -> abcd1234."""
    site = game.headers.get("Site", "")
    tail = site.rstrip("/").rsplit("/", 1)[-1]
    return tail or fallback


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_game(game: chess.pgn.Game, game_id: str) -> list[dict]:
    """Walk one game's mainline into per-move rows.

    A ply with no eval or no clock is skipped rather than stored as a null, so
    downstream code can trust both columns are present.
    """
    h = game.headers
    white, black = h.get("White", ""), h.get("Black", "")
    common = {
        "game_id": game_id,
        "white_elo": _int_or_none(h.get("WhiteElo")),
        "black_elo": _int_or_none(h.get("BlackElo")),
        "time_control": h.get("TimeControl", ""),
        "eco": h.get("ECO", ""),
        "opening": h.get("Opening", ""),
        "result": h.get("Result", ""),
        "termination": h.get("Termination", ""),
    }

    rows: list[dict] = []
    board = game.board()
    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        mover_is_white = board.turn == chess.WHITE
        fen_before = board.fen()
        san = board.san(move)

        score = node.eval()
        clock = node.clock()
        if score is not None and clock is not None:
            white_cp = score.white().score(mate_score=_MATE_CP)
            rows.append(
                {
                    **common,
                    "player_id": white if mover_is_white else black,
                    "color": "white" if mover_is_white else "black",
                    "ply": ply,
                    "move_number": (ply + 1) // 2,
                    "san": san,
                    "uci": move.uci(),
                    "fen_before": fen_before,
                    "eval_cp_white": white_cp,
                    "eval_cp": to_pov_cp(white_cp, mover_is_white),
                    "clock_s": clock,
                }
            )

        board.push(move)

    return rows


def parse_pgn_file(path: str | Path) -> Iterator[dict]:
    """Yield per-move rows for every game in a `.pgn` file."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        index = 0
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            game_id = _game_id(game, f"{path.stem}_{index}")
            yield from parse_game(game, game_id)
            index += 1


def parse_shard(shard_path: str | Path, out_path: str | Path) -> int:
    """Parse one shard to a parquet file, return the row count."""
    rows = list(parse_pgn_file(shard_path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)


def parse_interim(interim_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Parse every shard under interim_dir, one parquet each. Returns row counts."""
    interim_dir, out_dir = Path(interim_dir), Path(out_dir)
    counts: dict[str, int] = {}
    for shard in sorted(interim_dir.glob("shard_*.pgn")):
        out_path = out_dir / f"{shard.stem}.parquet"
        counts[shard.name] = parse_shard(shard, out_path)
    return counts
