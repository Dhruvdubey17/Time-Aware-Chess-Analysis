"""PART 4: run the whole classifier on a sample of complete games and review it.

Three stages (Maia inference runs in between, in the arm64 venv):

    python scripts/classify_sample.py sample     # pick games, build per-move rows, task lists
    python scripts/classify_sample.py engine      # Stockfish cps on the sample (30k nodes, mpv4)
    .venv_maia/bin/python scripts/maia_infer.py --tasks data/processed/classify_sample/maia_tasks.parquet --out data/processed/classify_sample/dispersion
    python scripts/classify_sample.py classify     # baseline vs time-aware, examples, divergence

Small on purpose (a few dozen complete games), enough to eyeball. It reuses the
STEP 1 think-time mapping and the Maia setup, and runs a little fresh Stockfish
on the sample only (same 30k-node, multipv-4 budget as the existing cache, so the
engine entropy fed to the think-time mapping matches its training). It does not
touch the existing cache, dataset, or mapping.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from chess_strength import classify as cl
from chess_strength import maia
from chess_strength import thinktime as tt
from chess_strength.complexity import complexity_features, decision_entropy
from chess_strength.config import load_config
from chess_strength.features import game_phase, material_cp
from chess_strength.stream_filter import classify_time_control
from chess_strength.timespent import parse_time_control

N_GAMES = 60
SEED = 0
MIN_PLIES = 20
ENGINE_NODES = 30000     # match the existing cache so engine entropy is comparable
ENGINE_MULTIPV = 4
ENGINE_WORKERS = 6
ENGINE_HASH_MB = 64
MATE_CP = 10000
DIR_NAME = "classify_sample"
# Glicko band cut points, same as the Tier-B sample, so bands match the mapping.
BANDS = [0, 1300, 1500, 1700, 1900, 2100, 9999]


def log(msg: str) -> None:
    print(msg, flush=True)


def _paths(cfg: dict):
    proc = Path(cfg["paths"]["processed"])
    return proc, proc / DIR_NAME


def _band_label(elo: float) -> str:
    return str(pd.cut([elo], BANDS, right=False)[0])


# ---------------------------------------------------------------------------
# stage: sample and build per-move rows
# ---------------------------------------------------------------------------

def _build_rows(g: pd.DataFrame, cfg: dict) -> list[dict]:
    """Per-move rows for one complete game, walked in ply order."""
    g = g.sort_values("ply")
    base, inc = parse_time_control(g.iloc[0]["time_control"])
    regime = classify_time_control(g.iloc[0]["time_control"])
    book_plies = cfg["book_plies"]
    prev_white_cp = 0  # startpos is about equal
    rows = []
    for r in g.itertuples():
        is_white = r.color == "white"
        before_cp = prev_white_cp if is_white else -prev_white_cp
        win_before = cl.win_pct_mate(before_cp)
        win_after = cl.win_pct_mate(r.eval_cp)
        elo = r.white_elo if is_white else r.black_elo
        band = _band_label(elo)
        rows.append({
            "game_id": r.game_id, "ply": r.ply, "color": r.color,
            "move_number": r.move_number, "san": r.san, "fen_before": r.fen_before,
            "regime": regime, "base": base, "inc": inc, "clock_s": r.clock_s,
            "win_before": win_before, "win_after": win_after,
            "wpl": max(0.0, win_before - win_after),
            "material_before": material_cp(r.fen_before, is_white),
            "is_capture": "x" in r.san,
            "in_book": r.ply <= book_plies,
            "game_phase": game_phase(r.fen_before, r.ply, book_plies),
            "player_rating": elo, "rating_band": band, "band_elo": maia.band_to_elo(band),
        })
        prev_white_cp = r.eval_cp_white
    return rows


def stage_sample(cfg: dict) -> None:
    _, out = _paths(cfg)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["game_id", "white_elo", "black_elo", "time_control", "color", "ply",
            "move_number", "san", "fen_before", "eval_cp_white", "eval_cp", "clock_s"]
    shard = min(glob.glob(str(Path(cfg["paths"]["interim"]) / "moves" / "*" / "shard_*.parquet")))
    df = pd.read_parquet(shard, columns=cols)

    # Keep blitz/rapid games with enough plies, then pick N by a fixed seed.
    tc_ok = df["time_control"].map(lambda t: classify_time_control(t) in ("blitz", "rapid"))
    df = df[tc_ok]
    counts = df.groupby("game_id")["ply"].size()
    eligible = counts[counts >= MIN_PLIES].index
    rng = np.random.default_rng(SEED)
    picked = rng.choice(np.array(eligible), size=min(N_GAMES, len(eligible)), replace=False)
    games = df[df["game_id"].isin(set(picked))]
    log(f"picked {len(picked)} complete games, {len(games):,} plies")

    rows = []
    for _, g in games.groupby("game_id"):
        rows.extend(_build_rows(g, cfg))
    moves = pd.DataFrame(rows)

    # Sacrifice size and think-time need the per-side sequence.
    moves = moves.sort_values(["game_id", "color", "ply"])
    grp = moves.groupby(["game_id", "color"], sort=False)
    prev_clock = grp["clock_s"].shift(1).fillna(moves["base"])
    moves["clock_before_s"] = prev_clock
    moves["time_spent_s"] = (prev_clock - moves["clock_s"] + moves["inc"]).clip(lower=0)
    # Material given up, mover POV, measured two of my own turns ahead so that a
    # piece won straight back within a ply or two does not read as a sacrifice.
    # Fall back to one turn ahead near the end of the game.
    settled = grp["material_before"].shift(-2).fillna(grp["material_before"].shift(-1))
    moves["sac_cp"] = (moves["material_before"] - settled).fillna(0.0).clip(lower=0)

    moves.to_parquet(out / "sample_moves.parquet", index=False)
    pd.DataFrame({"fen": moves["fen_before"].unique()}).to_parquet(out / "engine_tasks.parquet", index=False)
    tasks = moves[["fen_before", "regime", "band_elo"]].drop_duplicates().rename(columns={"fen_before": "fen"})
    tasks.to_parquet(out / "maia_tasks.parquet", index=False)
    log(f"sample_moves {len(moves):,} | engine tasks {moves['fen_before'].nunique():,} | "
        f"maia tasks {len(tasks):,}")
    log(f"regime mix: {moves['regime'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# stage: Stockfish cps on the sample (same proven worker pattern as the pilot)
# ---------------------------------------------------------------------------
_ENGINE = None


def _init_engine(sf_path: str, hash_mb: int) -> None:
    global _ENGINE
    import chess.engine

    _ENGINE = chess.engine.SimpleEngine.popen_uci(sf_path)
    _ENGINE.configure({"Threads": 1, "Hash": hash_mb})


def _cps_batch(fens: list[str]) -> list[dict]:
    from chess.engine import Limit

    out = []
    for fen in fens:
        board = chess.Board(fen)
        infos = _ENGINE.analyse(board, Limit(nodes=ENGINE_NODES), multipv=ENGINE_MULTIPV)
        cps = sorted((i["score"].relative.score(mate_score=MATE_CP) for i in infos), reverse=True)
        out.append({"fen": fen, "cps": cps, "n_legal": board.legal_moves.count()})
    return out


def stage_engine(cfg: dict) -> None:
    _, out = _paths(cfg)
    fens = pd.read_parquet(out / "engine_tasks.parquet")["fen"].tolist()
    log(f"scoring {len(fens):,} FENs with Stockfish at {ENGINE_NODES} nodes, multipv {ENGINE_MULTIPV}")
    batches = [fens[i:i + 100] for i in range(0, len(fens), 100)]
    start = time.time()
    results = []
    pool = mp.Pool(ENGINE_WORKERS, initializer=_init_engine,
                   initargs=(cfg["stockfish_path"], ENGINE_HASH_MB))
    try:
        for rows in pool.imap_unordered(_cps_batch, batches):
            results.extend(rows)
    finally:
        pool.terminate()
        pool.join()
    df = pd.DataFrame(results)
    # Engine entropy (the STEP 1 complexity) and the best-minus-second gap.
    df["engine_entropy"] = df["cps"].map(lambda cps: decision_entropy(list(cps)))
    df["eval_gap_1_2"] = df["cps"].map(
        lambda cps: complexity_features(list(cps), 99)["eval_gap_1_2"] or 0.0)
    df[["fen", "engine_entropy", "eval_gap_1_2", "n_legal"]].to_parquet(out / "engine_cps.parquet", index=False)
    log(f"done in {time.time() - start:.0f}s, {len(df):,} FENs")


# ---------------------------------------------------------------------------
# stage: classify and review
# ---------------------------------------------------------------------------

def _old_pressure(residual_s: float, clock_before_s: float) -> float:
    """The first-run pressure: max(fast, low-clock), where fast alone could create
    pressure on a full clock. Kept only to show the before/after of Change 3."""
    fast = min(1.0, max(0.0, -residual_s) / 10.0)
    low = min(1.0, max(0.0, 30.0 - clock_before_s) / 30.0)
    return max(fast, low)


def stage_classify(cfg: dict) -> None:
    proc, out = _paths(cfg)
    c = cl.ClassifyConfig.from_config(cfg)
    m = pd.read_parquet(out / "sample_moves.parquet")
    eng = pd.read_parquet(out / "engine_cps.parquet")
    disp = pd.concat((pd.read_parquet(p) for p in sorted(glob.glob(str(out / "dispersion" / "batch_*.parquet")))),
                     ignore_index=True)

    m = m.merge(eng, left_on="fen_before", right_on="fen", how="left")
    m = m.merge(disp[["fen", "regime", "band_elo", "maia_entropy"]],
                left_on=["fen_before", "regime", "band_elo"],
                right_on=["fen", "regime", "band_elo"], how="left")
    m = m.dropna(subset=["engine_entropy", "maia_entropy"]).reset_index(drop=True)
    log(f"classifying {len(m):,} moves with full features")

    # Expected think-time from the saved STEP 1 mapping (do not refit).
    model = tt.load_mapping(proc / "thinktime")
    md = m.rename(columns={"engine_entropy": "complexity"}).copy()
    md["move_number"] = md["move_number"].astype(float)
    md["log_clock_before"] = np.log1p(md["clock_before_s"].clip(lower=0))
    m["expected_think_s"] = tt.expected_thinktime(model, md)
    m["residual_s"] = m["time_spent_s"] - m["expected_think_s"]

    # Components and labels, row by row (small sample, clarity over speed). We also
    # keep the first-run pressure so the report can show the before/after of the
    # Change 3 clock gating on the same moves.
    def classify_row(r) -> pd.Series:
        gate = cl.complexity_gate(r.maia_entropy, c)
        pressure = cl.pressure_modifier(r.residual_s, r.clock_before_s, c)
        pressure_old = _old_pressure(r.residual_s, r.clock_before_s)
        skill = cl.skill_of_move(r.wpl, gate, pressure, c)
        skill_old = cl.skill_of_move(r.wpl, gate, pressure_old, c)
        brilliant = cl.is_brilliant(r.wpl, r.sac_cp, r.win_before, r.win_after, c)
        great = cl.is_great(r.wpl, r.eval_gap_1_2, c)
        base = cl.baseline_label(r.wpl, r.in_book, brilliant, great, c)
        return pd.Series({
            "gate": gate, "pressure": pressure, "pressure_old": pressure_old,
            "skill": skill, "brilliant_cand": brilliant, "great_cand": great,
            "baseline": base,
            "time_aware_B": cl.option_b_label(base, skill, brilliant, r.wpl, c),
            "time_aware_B_old": cl.option_b_label(base, skill_old, brilliant, r.wpl, c),
            "time_aware_A": cl.option_a_label(base, r.wpl, r.maia_entropy, pressure, c),
        })

    m = pd.concat([m, m.apply(classify_row, axis=1)], axis=1)
    m.to_parquet(out / "sample_labeled.parquet", index=False)

    _report(m, c, out)


def _report(m: pd.DataFrame, c: cl.ClassifyConfig, out: Path) -> None:
    upg = m[m["time_aware_B"] != m["baseline"]]
    log("\n=== Divergence (Option B) ===")
    log(f"  moves: {len(m):,}, changed label: {len(upg):,} ({len(upg) / len(m):.1%})")
    log(f"  Option A changed: {int((m['time_aware_A'] != m['baseline']).sum()):,}")
    log("  upgrades (baseline -> time-aware B):")
    for (b, t), n in upg.groupby(["baseline", "time_aware_B"]).size().sort_values(ascending=False).items():
        log(f"    {b:10s} -> {t:10s}  {n}")
    log("  upgrades by phase / regime / band:")
    for key in ["game_phase", "regime", "rating_band"]:
        frac = upg.groupby(key).size() / m.groupby(key).size()
        log(f"    {key}: " + ", ".join(f"{k}={v:.1%}" for k, v in frac.dropna().items()))

    # Baseline label mix and whether the special-label paths are even reachable.
    log("\n=== Baseline label mix ===")
    for lab, n in m["baseline"].value_counts().items():
        log(f"  {lab:11s} {n:5d}  ({n / len(m):.1%})")
    great_rate = (m["baseline"] == "Great").mean()
    log(f"  baseline Great rate {great_rate:.2%} (target < ~2%, gap {c.great_gap_cp:.0f}cp); "
        f"sac flagged {int((m['sac_cp'] >= c.brilliant_sac_cp).sum())}; "
        f"Brilliant {int(m['brilliant_cand'].sum())}; Great cand {int(m['great_cand'].sum())}")

    # Change 2 invariant: nothing below near-best is ever Great, in any output.
    for col in ["baseline", "time_aware_B", "time_aware_A"]:
        bad = int(((m[col] == "Great") & (m["wpl"] > c.near_best_max)).sum())
        log(f"  below-near-best Great in {col}: {bad}")

    # The guard: upgrade rate must be ~0 at low Maia dispersion.
    log("\n=== Guard: upgrade rate vs Maia dispersion (should be ~0 when easy) ===")
    m = m.copy()
    m["disp_bin"] = pd.qcut(m["maia_entropy"].rank(method="first"), 5, labels=False)
    for b, gb in m.groupby("disp_bin"):
        rate = (gb["time_aware_B"] != gb["baseline"]).mean()
        log(f"  dispersion quintile {int(b)} (mean {gb['maia_entropy'].mean():.2f}): "
            f"upgrade rate {rate:.1%}, n {len(gb):,}")

    # Change 3: upgrade rate vs absolute clock, first-run pressure vs now.
    log("\n=== Change 3: upgrade rate by absolute clock, OLD vs NEW pressure ===")
    edges = [0, 15, 30, 60, 120, 300, 1e12]
    names = ["<15s", "15-30s", "30-60s", "60-120s", "120-300s", "300s+"]
    m["clock_bin"] = pd.cut(m["clock_before_s"], edges, right=False, labels=names)
    for name in names:
        gb = m[m["clock_bin"] == name]
        if not len(gb):
            continue
        old = (gb["time_aware_B_old"] != gb["baseline"]).mean()
        new = (gb["time_aware_B"] != gb["baseline"]).mean()
        log(f"  {name:9s} n {len(gb):5d}  old {old:5.1%}  new {new:5.1%}")
    hi = m[m["clock_before_s"] >= 120]
    log(f"  high clock (>=120s) upgrades: old {int((hi['time_aware_B_old'] != hi['baseline']).sum())}, "
        f"new {int((hi['time_aware_B'] != hi['baseline']).sum())}")
    castle = m[(m["san"] == "O-O") & (m["clock_before_s"] >= 600)]
    if len(castle):
        r = castle.iloc[0]
        log(f"  full-clock castle check: O-O clock {r.clock_before_s:.0f}s -> "
            f"old {r.time_aware_B_old}, new {r.time_aware_B} (baseline {r.baseline})")

    _sensitivity(m, c)

    _examples(m, out)


def _sensitivity(m: pd.DataFrame, base: cl.ClassifyConfig) -> None:
    """How the upgrade rate moves with the gate floor and the Great cutoff.
    Pressure is unchanged, so we recompute only gate -> skill -> label."""
    from dataclasses import replace

    log("\n=== Sensitivity: overall upgrade rate (Great rate) ===")
    log("  rows = disp_lo (gate floor), cols = cut_great")
    cuts = [0.25, 0.35, 0.45]
    log("           " + "   ".join(f"cut={cg:.2f}" for cg in cuts))
    for lo in [0.70, 0.90, 1.10]:
        cells = []
        for cg in cuts:
            c2 = replace(base, disp_lo=lo, cut_great=cg)
            gate = m["maia_entropy"].map(lambda d, c2=c2: cl.complexity_gate(d, c2))
            skill = np.where(m["wpl"] <= c2.eligibility_max, gate * m["pressure"], 0.0)
            labels = [
                cl.option_b_label(b, s, bc, w, c2)
                for b, s, bc, w in zip(m["baseline"], skill, m["brilliant_cand"], m["wpl"])
            ]
            changed = np.mean([lab != b for lab, b in zip(labels, m["baseline"])])
            great = np.mean([lab == "Great" for lab in labels])
            cells.append(f"{changed:5.1%}({great:4.1%})")
        log(f"  lo={lo:.2f}  " + "  ".join(cells))


def _examples(m: pd.DataFrame, out: Path) -> None:
    def show(row) -> str:
        return (f"    {row.baseline}->{row.time_aware_B} | {row.san} ply{int(row.ply)} {row.regime} "
                f"band {row.rating_band}\n"
                f"      wpl {row.wpl:.1f}%  disp {row.maia_entropy:.2f}  gate {row.gate:.2f}  "
                f"press {row.pressure:.2f}  skill {row.skill:.2f}\n"
                f"      spent {row.time_spent_s:.1f}s vs expected {row.expected_think_s:.1f}s  "
                f"clock {row.clock_before_s:.0f}s  sac {row.sac_cp:.0f}cp\n"
                f"      {row.fen_before}")

    upg = m[m["time_aware_B"] != m["baseline"]]
    log("\n=== Example UPGRADES (eyeball whether they look justified) ===")
    for _, row in upg.sort_values("skill", ascending=False).head(6).iterrows():
        log(show(row))
    log("\n=== A few GOOD moves that stayed the same ===")
    same = m[(m["time_aware_B"] == m["baseline"]) & (m["wpl"] <= 5) & (~m["in_book"])]
    for _, row in same.sort_values("maia_entropy", ascending=False).head(3).iterrows():
        log(show(row))
    log("\n=== Near misses (eligible, hard, but skill just below the Great cut) ===")
    near = m[(m["skill"] > 0) & (m["skill"] < 0.35) & (m["time_aware_B"] == m["baseline"])]
    for _, row in near.sort_values("skill", ascending=False).head(3).iterrows():
        log(show(row))
    (out / "divergence.json").write_text(json.dumps({
        "n_moves": len(m),
        "upgraded_B": int((m["time_aware_B"] != m["baseline"]).sum()),
        "upgraded_A": int((m["time_aware_A"] != m["baseline"]).sum()),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "engine", "classify"])
    args = ap.parse_args()
    cfg = load_config()
    {"sample": stage_sample, "engine": stage_engine, "classify": stage_classify}[args.stage](cfg)


if __name__ == "__main__":
    main()
