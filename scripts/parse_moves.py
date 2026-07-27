"""Parse the Phase 2 shards into per-move parquet tables.

Usage:
    python scripts/parse_moves.py

Reads every shard_*.pgn under data/interim and writes one parquet per shard
into data/interim/moves. Prints a per-shard row count.
"""

from __future__ import annotations

import sys
from pathlib import Path

from chess_strength.config import load_config
from chess_strength.parse_moves import parse_interim


def main() -> int:
    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    out_dir = interim / "moves"

    shards = list(interim.glob("shard_*.pgn"))
    if not shards:
        print(f"no shards in {interim}; run scripts/filter_data.py first")
        return 1

    counts = parse_interim(interim, out_dir)
    total = sum(counts.values())
    for name, n in counts.items():
        print(f"{name}: {n} rows")
    print(f"total={total} rows -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
