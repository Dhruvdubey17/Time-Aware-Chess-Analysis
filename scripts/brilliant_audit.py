"""Audit the Brilliant (!!) baseline label across some games.

Runs the full review on each PGN and prints the Brilliant rate plus every move
that earns Brilliant with the facts behind it (Win% loss, Win% before the move,
the material sacrificed, and the Maia human-difficulty reading). Used to confirm
Brilliant is rare and each one is genuinely brilliant.

    python scripts/brilliant_audit.py tests/fixtures/bare.pgn other.pgn ...
    python scripts/brilliant_audit.py            # a default sample
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analyze import analyze
from backend.intake import parse_pgn_file
from chess_strength.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/private/tmp/claude-501/-Users-dhruvdubey-Files-PersonalProjects-"
               "TimedAwareChessAnalysis/a7ada617-7141-4c03-acbc-7bf05633656d/scratchpad")
DEFAULTS = [
    *sorted((SCRATCH / "real_games").glob("*.pgn")),
    ROOT / "tests" / "fixtures" / "bare.pgn",
    ROOT / "tests" / "fixtures" / "lichess_clk_eval.pgn",
]


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]] or DEFAULTS
    cfg = load_config()
    total_moves = total_brill = 0

    for path in args:
        if not path.exists():
            continue
        for rep in parse_pgn_file(path):
            if not rep.accepted:
                continue
            res = analyze(rep, cfg)
            brill = [m for m in res.moves if m["baseline_label"] == "Brilliant"]
            total_moves += len(res.moves)
            total_brill += len(brill)
            print(f"\n{path.name}: {rep.white} vs {rep.black}  "
                  f"{rep.regime} {rep.time_control}  moves {len(res.moves)}  "
                  f"BRILLIANT {len(brill)}")
            for m in brill:
                num = f"{m['move_number']}{'.' if m['side'] == 'white' else '...'}"
                print(f"  {num}{m['san']}  wpl {m['wpl']}  win_before {m['win_before']}  "
                      f"sac {m.get('sac_cp')}cp  disp {m['maia_entropy']} ({m['difficulty']})")

    rate = (total_brill / total_moves * 100) if total_moves else 0.0
    print(f"\nTOTAL Brilliant: {total_brill} / {total_moves} moves ({rate:.2f}%)")


if __name__ == "__main__":
    main()
