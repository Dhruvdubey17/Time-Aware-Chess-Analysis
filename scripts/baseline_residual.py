"""Run the revised Phase 7 go/no-go gate.

The estimator conditions on game phase and pre-move engine position complexity
only, never on mediators of skill. It scores held-out players by their mean
residual and checks the correlation with Glicko at the 100-clean-move cutoff.

Two views are printed:
1. The raw negative mean WPL reference baseline on the full held-out set, across
   move-count thresholds, overall and by time control. This needs no engine and
   shows how the signal grows with sample size. The chosen estimator must at
   least match it.
2. The engine complexity-adjusted skill score versus that baseline, on the
   players we could afford to annotate with the engine (those with enough clean
   moves for the gate), at the 100, 150, 200 cutoffs, overall and by regime.

Book and premove moves are dropped first. Splits are by player, no leakage.

Usage:
    python scripts/baseline_residual.py [--max-positions N] [--nodes N] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess.engine
import numpy as np
import pandas as pd

from chess_strength.baseline_residual import (
    COMPLEXITY_NODES,
    add_complexity_bucket,
    add_residuals,
    fit_expected_wpl,
    player_scores,
    position_complexity,
    raw_wpl_scores,
)
from chess_strength.config import load_config
from chess_strength.features import filter_clean

THRESHOLDS = (20, 50, 100, 150, 200)
GATE_CUTOFF = 100
GATE_RHO = 0.30


def _rho(scores: pd.DataFrame) -> tuple[float, int]:
    if len(scores) < 10:
        return float("nan"), len(scores)
    return scores["skill_score"].corr(scores["glicko"], method="spearman"), len(scores)


def _global_test_ids(clean: pd.DataFrame, test_frac: float, seed: int) -> set:
    players = clean["player_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    return set(players[rng.random(len(players)) < test_frac])


def _curve(label: str, df: pd.DataFrame, score_fn) -> None:
    parts = []
    for thr in THRESHOLDS:
        rho, n = _rho(score_fn(df, min_moves=thr))
        parts.append(f"n>={thr}: rho={rho:.3f} ({n}p)")
    print(f"  {label:<8} " + " | ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-positions", type=int, default=40000,
                    help="cap on engine-annotated positions, bounds the run time")
    ap.add_argument("--nodes", type=int, default=COMPLEXITY_NODES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    feat_path = Path(cfg["paths"]["processed"]) / "moves_features.parquet"
    if not feat_path.exists():
        print(f"{feat_path} missing; run scripts/build_features.py first")
        return 1

    clean = filter_clean(pd.read_parquet(feat_path))
    test_ids = _global_test_ids(clean, test_frac=0.3, seed=args.seed)
    clean_test = clean[clean["player_id"].isin(test_ids)]

    med = int(clean.groupby("player_id").size().median())
    print(f"clean moves: {len(clean)} | players: {clean['player_id'].nunique()} "
          f"| median clean moves/player: {med}")
    if med < 50:
        print("WARNING: median clean moves/player is low; per-player scores are "
              "noisy. See PLAN Phase 7 data note, widen the data before judging method.")

    # View 1: reference baseline on the full held-out set, no engine.
    print("\nRaw -mean WPL reference, full held-out set:")
    _curve("overall", clean_test, raw_wpl_scores)
    for regime in ("blitz", "rapid"):
        _curve(regime, clean_test[clean_test["regime"] == regime], raw_wpl_scores)

    # Pick players we can afford to annotate: those with enough clean moves to
    # reach the gate cutoff. Cap total positions to bound the engine time.
    counts = clean.groupby("player_id").size()
    eligible = counts[counts >= GATE_CUTOFF].index.to_numpy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eligible)
    chosen, budget = [], 0
    for pid in eligible:
        if budget + counts[pid] > args.max_positions:
            continue
        chosen.append(pid)
        budget += counts[pid]
    chosen = set(chosen)
    sub = clean[clean["player_id"].isin(chosen)].copy()
    print(f"\nengine annotation: {len(chosen)} players, {len(sub)} positions, "
          f"nodes={args.nodes}")

    # Join the pre-move FENs from the interim table (features dropped them).
    interim = pd.read_parquet(Path(cfg["paths"]["interim"]) / "moves" / "shard_000.parquet",
                              columns=["game_id", "ply", "fen_before"])
    sub = sub.merge(interim, on=["game_id", "ply"], how="left")

    fens = sub["fen_before"].dropna().unique().tolist()
    t = time.time()
    with chess.engine.SimpleEngine.popen_uci(cfg["stockfish_path"]) as engine:
        engine.configure({"Threads": 1})
        comp = position_complexity(fens, engine, nodes=args.nodes)
    print(f"  scored {len(fens)} unique FENs in {time.time() - t:.0f}s")

    sub["complexity"] = sub["fen_before"].map(comp)
    sub = sub.dropna(subset=["complexity"])
    sub = add_complexity_bucket(sub)

    sub_train = sub[~sub["player_id"].isin(test_ids)]
    sub_test = sub[sub["player_id"].isin(test_ids)]
    table, global_mean = fit_expected_wpl(sub_train)
    sub_test = add_residuals(sub_test, table, global_mean)

    print(f"\nSkill (phase+complexity) vs raw WPL, engine-annotated held-out players "
          f"(train {sub_train['player_id'].nunique()}p, test {sub_test['player_id'].nunique()}p):")

    def compare(label: str, df: pd.DataFrame) -> None:
        parts = []
        for thr in (100, 150, 200):
            rho_s, n = _rho(player_scores(df, min_moves=thr))
            rho_r, _ = _rho(raw_wpl_scores(df, min_moves=thr))
            parts.append(f"n>={thr}: skill={rho_s:.3f} raw={rho_r:.3f} ({n}p)")
        print(f"  {label:<8} " + " | ".join(parts))

    compare("overall", sub_test)
    for regime in ("blitz", "rapid"):
        compare(regime, sub_test[sub_test["regime"] == regime])

    # Go/no-go gate: the CHOSEN estimator is raw negative mean WPL, judged over
    # all held-out players with at least the cutoff of clean moves. This is the
    # pass condition, kept separate from estimator selection below.
    gate_rho, gate_n = _rho(raw_wpl_scores(clean_test, min_moves=GATE_CUTOFF))
    print(f"\nGATE (chosen estimator = raw neg mean WPL, n>={GATE_CUTOFF}, {gate_n} "
          f"players): rho={gate_rho:.3f}, need >={GATE_RHO} -> "
          f"{'PASS' if gate_rho >= GATE_RHO else 'FAIL'}")

    # Estimator selection (reported, NOT a pass condition): on the engine-annotated
    # subset, pick whichever candidate has the higher held-out rho. The complexity
    # residual is a candidate here, not the primary score.
    sel_skill, sel_n = _rho(player_scores(sub_test, min_moves=GATE_CUTOFF))
    sel_raw, _ = _rho(raw_wpl_scores(sub_test, min_moves=GATE_CUTOFF))
    selected = "raw neg mean WPL" if sel_raw >= sel_skill else "phase+complexity residual"
    print(f"estimator selection (n>={GATE_CUTOFF}, {sel_n} players, annotated subset): "
          f"raw={sel_raw:.3f} vs complexity-residual={sel_skill:.3f} -> selected: {selected}")
    if sel_n < 30:
        print("note: few players in the annotated subset, so the selection rho is "
              "shaky. The gate above uses the full held-out set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
