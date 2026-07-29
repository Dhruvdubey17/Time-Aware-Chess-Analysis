"""Checkpoint A: run the full backend on real games and show the upgrades.

Real Lichess games are already on disk from the earlier work
(data/interim/moves). The earlier classifier flagged which of a 60-game sample
got a time-aware upgrade; we rebuild those games as PGN and run them through the
NEW backend to confirm it reproduces genuine upgrades, then print the facts a
chess player needs to judge each one.

    python scripts/checkpoint_a.py            # 3 real upgraded games
    python scripts/checkpoint_a.py --n 5

Reconstructed PGNs are written to the scratchpad so they can be pasted into the
app to reproduce.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analyze import analyze
from backend.intake import parse_pgn
from chess_strength.config import load_config

LABELED = "data/processed/classify_sample/sample_labeled.parquet"
OUT_DIR = Path("/private/tmp/claude-501/-Users-dhruvdubey-Files-PersonalProjects-"
               "TimedAwareChessAnalysis/a7ada617-7141-4c03-acbc-7bf05633656d/"
               "scratchpad/real_games")


def _to_clk(sec: float) -> str:
    s = round(float(sec))  # float() so a numpy clock value rounds to a plain int
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _to_eval(cp: int) -> str:
    return f"{cp / 100:.2f}"


def rebuild_pgn(rows: pd.DataFrame) -> str:
    """A real game rebuilt into PGN with its clocks and evals, via python-chess
    so the movetext and numbering are always valid."""
    rows = rows.sort_values("ply")
    r0 = rows.iloc[0]
    white = rows[rows.color == "white"]["player_id"].iloc[0]
    black = rows[rows.color == "black"]["player_id"].iloc[0]
    game = chess.pgn.Game()
    game.headers.update({
        "Event": "Rated Blitz game", "Site": f"https://lichess.org/{r0.game_id}",
        "White": str(white), "Black": str(black), "Result": r0.result,
        "WhiteElo": str(r0.white_elo), "BlackElo": str(r0.black_elo),
        "TimeControl": r0.time_control, "ECO": r0.eco, "Opening": r0.opening,
        "Termination": r0.termination,
    })
    node = game
    for r in rows.itertuples():
        node = node.add_variation(chess.Move.from_uci(r.uci))
        node.comment = f"[%eval {_to_eval(r.eval_cp_white)}] [%clk {_to_clk(r.clock_s)}]"
    return str(game) + "\n"


def show(report, result) -> None:
    print(f"\n{'=' * 78}")
    print(f"{report.white} ({report.white_elo}) vs {report.black} ({report.black_elo})"
          f"   {report.result}   {report.opening}")
    print(f"  {report.site}  {report.regime}  {report.time_control}  "
          f"{report.n_moves} moves   {report.capability_note[:70]}")
    s = result.summary
    print(f"  baseline: {s['baseline_counts']}")
    if s["time_aware_counts"]:
        print(f"  time-aware: {s['time_aware_counts']}")
    print(f"  pressure upgrades: {s['n_upgrades']}")
    for u in s["upgrades"]:
        num = f"{u['move_number']}.{'' if u['side'] == 'white' else '..'}{u['san']}"
        print(f"\n  UPGRADE  {num}   {u['baseline']} -> {u['time_aware']}")
        print(f"    Win% lost by the move : {u['wpl']:.1f}")
        print(f"    how hard for a human  : {u['difficulty']} "
              f"(Maia dispersion {u['maia_entropy']})")
        print(f"    time spent vs expected: {u['time_spent_s']:.0f}s vs "
              f"{u['expected_think_s']:.0f}s")
        print(f"    clock left            : {u['clock_before_s']:.0f}s")
        print(f"    position (FEN)        : {u['fen_before']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labeled = pd.read_parquet(LABELED)
    upgraded_ids = list(labeled[labeled.time_aware_B != labeled.baseline]["game_id"].unique())
    print(f"real games the earlier classifier upgraded: {len(upgraded_ids)}")

    shard = min(glob.glob("data/interim/moves/*/shard_*.parquet"))
    cols = ["game_id", "white_elo", "black_elo", "time_control", "eco", "opening",
            "result", "termination", "player_id", "color", "ply", "uci",
            "eval_cp_white", "clock_s"]
    interim = pd.read_parquet(shard, columns=cols)
    interim = interim[interim.game_id.isin(upgraded_ids[: args.n])]

    for gid in upgraded_ids[: args.n]:
        rows = interim[interim.game_id == gid]
        pgn = rebuild_pgn(rows)
        (OUT_DIR / f"{gid}.pgn").write_text(pgn)
        report = parse_pgn(pgn)[0]
        result = analyze(report, cfg)
        show(report, result)

    print(f"\nreconstructed PGNs saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
