"""Show the Phase 1 intake on any PGN, or on the test fixtures by default.

    python scripts/intake_demo.py                       # all fixtures
    python scripts/intake_demo.py path/to/game.pgn      # your own file

Prints the honest capability report for each game and the first few normalized
moves, so you can see what the backend will hand the later phases.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.intake import GameReport, parse_pgn_file

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FIXTURE_FILES = ["lichess_clk_eval.pgn", "chesscom_clk.pgn", "bare.pgn",
                 "incomplete.pgn", "variant.pgn", "mini.pgn"]


def show(report: GameReport) -> None:
    if not report.accepted:
        print(f"  [game {report.index}] REJECTED: {report.reject_reason}")
        return

    print(f"  [game {report.index}] {report.white} vs {report.black}  "
          f"{report.result}  ({report.opening or 'opening unknown'})")
    print(f"      site={report.site}  time_control={report.time_control or '-'}  "
          f"regime={report.regime}  moves={report.n_moves}")
    print(f"      clocks={report.has_clocks}  evals={report.has_evals}  "
          f"time_aware={report.time_aware_available}")
    print(f"      note: {report.capability_note}")
    for m in report.moves[:3]:
        clk = f"{m.clock_s:.0f}s" if m.clock_s is not None else "-"
        ev = m.eval_cp_white if m.eval_cp_white is not None else "-"
        print(f"        ply {m.ply:>2} {m.side:<5} {m.san:<6} clock={clk:<6} "
              f"eval_white={ev}  {m.phase}{' book' if m.in_book else ''}")


def main() -> None:
    args = sys.argv[1:]
    files = [Path(a) for a in args] if args else [FIXTURES / f for f in FIXTURE_FILES]
    for path in files:
        print(f"\n=== {path.name} ===")
        reports = parse_pgn_file(path)
        if not reports:
            print("  no games found")
        for report in reports:
            show(report)


if __name__ == "__main__":
    main()
