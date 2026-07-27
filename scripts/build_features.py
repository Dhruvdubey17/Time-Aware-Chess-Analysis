"""Assemble the Phase 6 feature tables from parsed per-move parquet.

Usage:
    python scripts/build_features.py

Reads every parquet under data/interim/moves, joins the per-move signals, and
writes moves_features.parquet and players.parquet under data/processed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from chess_strength.config import load_config
from chess_strength.features import build_and_write


def main() -> int:
    cfg = load_config()
    moves_dir = Path(cfg["paths"]["interim"]) / "moves"
    shards = sorted(moves_dir.glob("*.parquet"))
    if not shards:
        print(f"no parquet in {moves_dir}; run scripts/parse_moves.py first")
        return 1

    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    rows = df.to_dict("records")

    out_dir = Path(cfg["paths"]["processed"])
    counts = build_and_write(rows, cfg, out_dir)
    print(f"moves={counts['moves']} players={counts['players']} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
