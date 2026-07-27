"""Filter one PGN source (fixture or a real monthly .pgn.zst) into shards.

Usage:
    python scripts/filter_data.py [SRC]

SRC defaults to the test fixture so the pipeline is runnable out of the box.
Pass a downloaded data/raw/*.pgn.zst to filter a real month. Kept games are
written to data/interim/ and a one-line stats summary is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from chess_strength.config import load_config
from chess_strength.stream_filter import run_filter

DEFAULT_SRC = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mini.pgn"


def main() -> int:
    cfg = load_config()
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.exists():
        print(f"source not found: {src}")
        return 1

    out_dir = Path(cfg["paths"]["interim"])
    stats = run_filter(src, out_dir, cfg["regimes"])
    print(
        f"seen={stats.games_seen} kept={stats.games_kept} "
        f"keep_rate={stats.keep_rate:.4f} shards={stats.shards_written} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
