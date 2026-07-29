"""Run the full backend review on a game and print it in plain text.

    python scripts/review_demo.py tests/fixtures/lichess_clk_eval.pgn
    python scripts/review_demo.py my_games.pgn --game 2 --option A

Shows the capability report, the baseline vs time-aware label for every move,
and the supporting facts for each pressure upgrade. This is the view used to
sanity-check the labels before any UI exists.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analyze import analyze
from backend.intake import parse_pgn_file
from chess_strength.config import load_config


def _progress(stage: str, done: int, total: int) -> None:
    print(f"\r  [{stage}] {done}/{total}", end="", flush=True)
    if done >= total:
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", type=Path)
    ap.add_argument("--game", type=int, default=0, help="0-based game index in the file")
    ap.add_argument("--option", choices=["A", "B"], default="B")
    args = ap.parse_args()

    cfg = load_config()
    reports = parse_pgn_file(args.pgn)
    if not reports:
        print("no games found")
        return
    report = reports[args.game]
    if not report.accepted:
        print(f"REJECTED: {report.reject_reason}")
        return

    print(f"Game: {report.white} vs {report.black}  {report.result}  "
          f"({report.opening or 'opening unknown'})")
    print(f"  {report.site}  {report.regime}  {report.time_control}  "
          f"{report.n_moves} moves")
    print(f"  {report.capability_note}")
    print("analyzing ...")

    start = time.time()
    result = analyze(report, cfg, option=args.option, progress=_progress)
    print(f"done in {time.time() - start:.1f}s\n")

    s = result.summary
    print(f"time-aware: {'available' if result.time_aware_available else 'unavailable'}")
    print(f"baseline labels: {s['baseline_counts']}")
    if s["time_aware_counts"]:
        print(f"time-aware labels: {s['time_aware_counts']}")
    print(f"pressure upgrades: {s['n_upgrades']}\n")

    for u in s["upgrades"]:
        num = f"{u['move_number']}.{'' if u['side'] == 'white' else '..'}"
        spent = f"{u['time_spent_s']:.0f}s" if u["time_spent_s"] is not None else "-"
        exp = f"{u['expected_think_s']:.0f}s" if u["expected_think_s"] is not None else "-"
        clk = f"{u['clock_before_s']:.0f}s" if u["clock_before_s"] is not None else "-"
        print(f"  UPGRADE {num}{u['san']}  {u['baseline']} -> {u['time_aware']}")
        print(f"    Win% loss {u['wpl']:.1f}   difficulty {u['difficulty']} "
              f"(dispersion {u['maia_entropy']})")
        print(f"    spent {spent} vs expected {exp}   clock left {clk}")
        print(f"    {u['fen_before']}")

    print("\nall moves:")
    for m in result.moves:
        num = f"{m['move_number']}.{'' if m['side'] == 'white' else '..'}"
        ta = m["time_aware_label"] or "-"
        flag = "  <== upgrade" if m["upgraded"] else ""
        disp = f"{m['maia_entropy']}" if m["maia_entropy"] is not None else "-"
        print(f"  {num:>6}{m['san']:<7} {m['baseline_label']:<10} -> {ta:<10} "
              f"wpl {m['wpl']:5.1f}  disp {disp:<6}{flag}")


if __name__ == "__main__":
    main()
