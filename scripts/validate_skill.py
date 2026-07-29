"""STEP 5: validate that time-aware upgrades track genuine skill. Rating-free.

Rating never enters the classifier. Maia dispersion is scored at a FIXED reference
Elo (1500) and the think-time mapping is fed a FIXED reference band, so complexity
and time pressure carry no rating. Regime (time control) still routes the Maia
model, which is not rating. Glicko and future rating are held out for validation.

Stages (Maia runs in between, arm64 venv):
    python scripts/validate_skill.py sample     # player- and time-disjoint games, per-move rows
    python scripts/validate_skill.py engine       # Stockfish entropy + gap on the sample
    .venv_maia/bin/python scripts/maia_infer.py --tasks data/processed/skill_val/maia_tasks.parquet --out data/processed/skill_val/dispersion
    python scripts/validate_skill.py classify      # labels at both cutoffs, the three tests, verdict

Reuses the dataset, the revised classify.py, and the STEP 1 mapping. Nothing is
re-downloaded, rebuilt, or deleted.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from chess_strength import classify as cl
from chess_strength import thinktime as tt
from chess_strength import validate as val
from chess_strength.complexity import complexity_features, decision_entropy
from chess_strength.config import load_config
from chess_strength.features import game_phase, material_cp
from chess_strength.stream_filter import classify_time_control
from chess_strength.timespent import parse_time_control

SEED = 0
TARGET_GAMES = 5000          # complete games; scaled to get thousands of upgrades at 120s
CAP_GAMES_PER_PLAYER = 8
FIXED_ELO = 1500             # rating-free Maia reference
FIXED_BAND = "[1500, 1700)"  # rating-free think-time reference band
CLOCK_CUTOFFS = [60, 120]
ENGINE_NODES = 30000
ENGINE_MULTIPV = 4
ENGINE_WORKERS = 6
ENGINE_HASH_MB = 64
MATE_CP = 10000
MIN_MOVES_PER_PLAYER = 20    # floor for a stable per-player upgrade rate
DIR_NAME = "skill_val"
INTERIM_COLS = ["game_id", "white_elo", "black_elo", "time_control", "player_id",
                "color", "ply", "move_number", "san", "fen_before", "eval_cp_white", "eval_cp"]
CLOCK_COL = "clock_s"


def log(msg: str) -> None:
    print(msg, flush=True)


def _paths(cfg: dict):
    proc = Path(cfg["paths"]["processed"])
    return proc, proc / DIR_NAME


def _excluded_players(cfg: dict, proc: Path) -> set:
    """Players used to tune thresholds (the classify_sample games) or in the Tier-B
    sample. The STEP 5 test set is disjoint from these."""
    excl = set()
    sp = proc / "tierb" / "sample_players.parquet"
    if sp.exists():
        excl |= set(pd.read_parquet(sp)["player_id"])
    tuning = proc / "classify_sample" / "sample_moves.parquet"
    if tuning.exists():
        tuning_games = set(pd.read_parquet(tuning)["game_id"])
        # Map those games to their players via the interim table.
        for shard in glob.glob(str(Path(cfg["paths"]["interim"]) / "moves" / "*" / "shard_*.parquet")):
            d = pd.read_parquet(shard, columns=["game_id", "player_id"])
            excl |= set(d[d["game_id"].isin(tuning_games)]["player_id"])
    return excl


# ---------------------------------------------------------------------------
# stage: sample player- and time-disjoint complete games
# ---------------------------------------------------------------------------

def _player_game_index(cfg: dict) -> pd.DataFrame:
    """Compact (player_id, game_id, month) table over the whole dataset."""
    frames = []
    for shard in sorted(glob.glob(str(Path(cfg["paths"]["interim"]) / "moves" / "*" / "shard_*.parquet"))):
        month = Path(shard).parent.name
        d = pd.read_parquet(shard, columns=["player_id", "game_id"]).drop_duplicates()
        d["month"] = month
        frames.append(d)
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _build_rows(g: pd.DataFrame, cfg: dict, month: str) -> list[dict]:
    g = g.sort_values("ply")
    base, inc = parse_time_control(g.iloc[0]["time_control"])
    regime = classify_time_control(g.iloc[0]["time_control"])
    book_plies = cfg["book_plies"]
    prev_white_cp = 0
    rows = []
    for r in g.itertuples():
        is_white = r.color == "white"
        before_cp = prev_white_cp if is_white else -prev_white_cp
        win_before = cl.win_pct_mate(before_cp)
        win_after = cl.win_pct_mate(r.eval_cp)
        rows.append({
            "game_id": r.game_id, "ply": r.ply, "color": r.color,
            "player_id": r.player_id, "month": month, "regime": regime,
            "move_number": r.move_number, "san": r.san, "fen_before": r.fen_before,
            "base": base, "inc": inc, "clock_s": getattr(r, CLOCK_COL),
            "win_before": win_before, "win_after": win_after,
            "wpl": max(0.0, win_before - win_after),
            "material_before": material_cp(r.fen_before, is_white),
            "is_capture": "x" in r.san,
            "in_book": r.ply <= book_plies,
            "game_phase": game_phase(r.fen_before, r.ply, book_plies),
            # Held out for validation only, never fed to the classifier.
            "glicko": r.white_elo if is_white else r.black_elo,
        })
        prev_white_cp = r.eval_cp_white
    return rows


def stage_sample(cfg: dict) -> None:
    proc, out = _paths(cfg)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    excl = _excluded_players(cfg, proc)
    log(f"excluded players (Tier-B + tuning): {len(excl):,}")
    pg = _player_game_index(cfg)
    log(f"player-game index rows: {len(pg):,}")

    # Eligible players: appear in both months (for the temporal test) and not excluded.
    months_per = pg.groupby("player_id")["month"].nunique()
    both = set(months_per[months_per >= 2].index) - excl
    log(f"eligible players (both months, disjoint): {len(both):,}")

    # Greedily gather games, capped per player, until we hit the target.
    by_player = {}
    for pid, gid in zip(pg["player_id"].to_numpy(), pg["game_id"].to_numpy()):
        if pid in both:
            by_player.setdefault(pid, []).append(gid)
    order = list(both)
    rng.shuffle(order)
    chosen = set()
    for pid in order:
        for gid in by_player[pid][:CAP_GAMES_PER_PLAYER]:
            chosen.add(gid)
        if len(chosen) >= TARGET_GAMES:
            break
    log(f"chosen games: {len(chosen):,}")

    # Second pass: pull those complete games and build per-move rows.
    rows = []
    for shard in sorted(glob.glob(str(Path(cfg["paths"]["interim"]) / "moves" / "*" / "shard_*.parquet"))):
        month = Path(shard).parent.name
        d = pd.read_parquet(shard, columns=INTERIM_COLS + [CLOCK_COL])
        d = d[d["game_id"].isin(chosen)]
        for _, g in d.groupby("game_id"):
            rows.extend(_build_rows(g, cfg, month))
    moves = pd.DataFrame(rows)

    # Think-time and sacrifice, per side, in ply order.
    moves = moves.sort_values(["game_id", "color", "ply"])
    grp = moves.groupby(["game_id", "color"], sort=False)
    prev_clock = grp["clock_s"].shift(1).fillna(moves["base"])
    moves["clock_before_s"] = prev_clock
    moves["time_spent_s"] = (prev_clock - moves["clock_s"] + moves["inc"]).clip(lower=0)
    settled = grp["material_before"].shift(-2).fillna(grp["material_before"].shift(-1))
    moves["sac_cp"] = (moves["material_before"] - settled).fillna(0.0).clip(lower=0)
    moves["eligible"] = ~moves["player_id"].isin(excl)

    moves.to_parquet(out / "sample_moves.parquet", index=False)
    # Rating-free scoring tasks: unique FEN, and unique FEN x regime at the fixed Elo.
    pd.DataFrame({"fen": moves["fen_before"].unique()}).to_parquet(out / "engine_tasks.parquet", index=False)
    tasks = moves[["fen_before", "regime"]].drop_duplicates().rename(columns={"fen_before": "fen"})
    tasks["band_elo"] = FIXED_ELO
    tasks.to_parquet(out / "maia_tasks.parquet", index=False)

    post = moves[~moves["in_book"]]
    log(f"moves {len(moves):,} (post-book {len(post):,}); unique FENs {moves['fen_before'].nunique():,}")
    log(f"eligible movers {moves['eligible'].sum():,}; players {moves.loc[moves['eligible'],'player_id'].nunique():,}")
    log(f"engine tasks {moves['fen_before'].nunique():,}; maia tasks {len(tasks):,}; "
        f"regime mix {moves['regime'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# stage: Stockfish entropy + gap (same proven worker pattern)
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
        out.append({"fen": fen, "engine_entropy": decision_entropy(cps),
                    "eval_gap_1_2": (complexity_features(cps, 99)["eval_gap_1_2"] or 0.0)})
    return out


def _flush_engine(rows: list[dict], cache_dir: Path, idx: int) -> None:
    tmp = cache_dir / f".batch_{idx:05d}.tmp.parquet"
    final = cache_dir / f"batch_{idx:05d}.parquet"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, final)


def _engine_done(cache_dir: Path) -> set:
    done = set()
    for b in sorted(cache_dir.glob("batch_*.parquet")):
        done |= set(pd.read_parquet(b, columns=["fen"])["fen"])
    return done


def stage_engine(cfg: dict) -> None:
    _, out = _paths(cfg)
    cache_dir = out / "engine_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = _engine_done(cache_dir)
    fens = [f for f in pd.read_parquet(out / "engine_tasks.parquet")["fen"] if f not in done]
    log(f"Stockfish on {len(fens):,} FENs ({len(done):,} already done) at {ENGINE_NODES} nodes")
    if not fens:
        return
    batches = [fens[i:i + 200] for i in range(0, len(fens), 200)]
    start_idx = len(list(cache_dir.glob("batch_*.parquet")))
    start = time.time()
    total = 0
    pool = mp.Pool(ENGINE_WORKERS, initializer=_init_engine,
                   initargs=(cfg["stockfish_path"], ENGINE_HASH_MB))
    try:
        for k, rows in enumerate(pool.imap_unordered(_cps_batch, batches)):
            _flush_engine(rows, cache_dir, start_idx + k)   # atomic, resumable
            total += len(rows)
            if (k + 1) % 20 == 0:
                el = time.time() - start
                log(f"  {total:,}/{len(fens):,}  {el/60:.1f} min  {total/el*60:.0f}/min")
    finally:
        pool.terminate()
        pool.join()
    el = time.time() - start
    log(f"done: {total:,} new in {el/60:.1f} min ({total/max(el,1)*60:.0f}/min), "
        f"{ENGINE_WORKERS} workers; cache now {len(_engine_done(cache_dir)):,} FENs")


# ---------------------------------------------------------------------------
# stage: classify at both cutoffs and run the three tests
# ---------------------------------------------------------------------------

def _label_at_cutoff(m: pd.DataFrame, base_c: cl.ClassifyConfig, cutoff: int) -> pd.DataFrame:
    from dataclasses import replace

    c = replace(base_c, clock_danger_s=float(cutoff))
    gate = m["maia_entropy"].map(lambda d: cl.complexity_gate(d, c))
    pressure = [cl.pressure_modifier(r, cb, c) for r, cb in zip(m["residual_s"], m["clock_before_s"])]
    pressure = np.array(pressure)
    skill = np.where(m["wpl"].to_numpy() <= c.eligibility_max, gate.to_numpy() * pressure, 0.0)
    b_labels, a_labels = [], []
    for base, s, br, w, disp, pr in zip(m["baseline"], skill, m["brilliant_cand"], m["wpl"],
                                        m["maia_entropy"], pressure):
        b_labels.append(cl.option_b_label(base, s, br, w, c))
        a_labels.append(cl.option_a_label(base, w, disp, pr, c))
    res = pd.DataFrame({f"B_{cutoff}": b_labels, f"A_{cutoff}": a_labels})
    res[f"upB_{cutoff}"] = res[f"B_{cutoff}"].to_numpy() != m["baseline"].to_numpy()
    res[f"upA_{cutoff}"] = res[f"A_{cutoff}"].to_numpy() != m["baseline"].to_numpy()
    return res


def stage_classify(cfg: dict) -> None:
    proc, out = _paths(cfg)
    c = cl.ClassifyConfig.from_config(cfg)
    m = pd.read_parquet(out / "sample_moves.parquet")
    eng = pd.concat((pd.read_parquet(p) for p in sorted(glob.glob(str(out / "engine_cache" / "batch_*.parquet")))),
                    ignore_index=True).drop_duplicates("fen")
    disp = pd.concat((pd.read_parquet(p) for p in sorted(glob.glob(str(out / "dispersion" / "batch_*.parquet")))),
                     ignore_index=True)
    # Do not run the tests on a partial engine or Maia pass.
    n_tasks = len(pd.read_parquet(out / "engine_tasks.parquet"))
    if len(eng) < n_tasks or len(disp) < len(pd.read_parquet(out / "maia_tasks.parquet")):
        log(f"INCOMPLETE: engine {len(eng):,}/{n_tasks:,}, maia {len(disp):,}. Finish those stages first.")
        raise SystemExit(1)
    m = m.merge(eng, left_on="fen_before", right_on="fen", how="left")
    m = m.merge(disp[["fen", "regime", "maia_entropy"]].rename(columns={"fen": "fen_d"}),
                left_on=["fen_before", "regime"], right_on=["fen_d", "regime"], how="left")
    m = m.dropna(subset=["engine_entropy", "maia_entropy"]).reset_index(drop=True)
    log(f"classifying {len(m):,} moves ({m['eligible'].sum():,} eligible movers)")

    # Expected think-time from the saved mapping, fed a FIXED band (rating-free).
    model = tt.load_mapping(proc / "thinktime")
    md = pd.DataFrame({
        "complexity": m["engine_entropy"], "move_number": m["move_number"].astype(float),
        "log_clock_before": np.log1p(m["clock_before_s"].clip(lower=0)),
        "game_phase": m["game_phase"], "regime": m["regime"], "rating_band": FIXED_BAND,
    })
    m["residual_s"] = m["time_spent_s"] - tt.expected_thinktime(model, md)

    # Baseline (time-blind, complexity-blind, rating-blind).
    m["brilliant_cand"] = [cl.is_brilliant(w, s, wb, wa, c)
                           for w, s, wb, wa in zip(m["wpl"], m["sac_cp"], m["win_before"], m["win_after"])]
    m["great_cand"] = [cl.is_great(w, g, c) for w, g in zip(m["wpl"], m["eval_gap_1_2"])]
    m["baseline"] = [cl.baseline_label(w, ib, br, gr, c)
                     for w, ib, br, gr in zip(m["wpl"], m["in_book"], m["brilliant_cand"], m["great_cand"])]

    for cutoff in CLOCK_CUTOFFS:
        m = pd.concat([m, _label_at_cutoff(m, c, cutoff)], axis=1)
    m.to_parquet(out / "labeled.parquet", index=False)

    report = {"n_moves": len(m), "n_eligible": int(m["eligible"].sum()),
              "fixed_elo": FIXED_ELO, "fixed_band": FIXED_BAND, "cutoffs": {}}
    for cutoff in CLOCK_CUTOFFS:
        report["cutoffs"][str(cutoff)] = _run_tests(m, cutoff)

    report["clock_recommendation"] = _recommend(report["cutoffs"])
    (out / "skill_report.json").write_text(json.dumps(report, indent=2, default=str))
    _print_report(report)
    log(f"\nsaved report: {out / 'skill_report.json'}")


def _players_frame(m: pd.DataFrame, up_col: str) -> pd.DataFrame:
    """Per eligible player: upgrade rate, glicko, dominant regime, month splits."""
    e = m[m["eligible"] & ~m["in_book"]].copy()
    g = e.groupby("player_id")
    players = g.agg(
        upgrade_rate=(up_col, "mean"), n_moves=(up_col, "size"),
        glicko=("glicko", "mean"),
        regime=("regime", lambda s: s.mode().iloc[0] if len(s.mode()) else "blitz"),
    )
    apr = e[e["month"] == "2017-04"].groupby("player_id").agg(
        apr_up=(up_col, "mean"), apr_wpl=("wpl", "mean"), apr_n=(up_col, "size"))
    may = e[e["month"] == "2017-05"].groupby("player_id").agg(
        may_glicko=("glicko", "mean"), may_n=("glicko", "size"))
    return players.join(apr).join(may).reset_index()


def _run_tests(m: pd.DataFrame, cutoff: int) -> dict:
    out = {}
    for opt, up_col in [("B", f"upB_{cutoff}"), ("A", f"upA_{cutoff}")]:
        e = m[m["eligible"] & ~m["in_book"]]
        n_events = int(e[up_col].sum())
        players = _players_frame(m, up_col)
        stable = players[players["n_moves"] >= MIN_MOVES_PER_PLAYER]

        t1 = val.rating_monotonicity(stable["upgrade_rate"].to_numpy(), stable["glicko"].to_numpy())
        t1_by = {}
        for reg, gp in stable.groupby("regime"):
            t1_by[reg] = val.rating_monotonicity(gp["upgrade_rate"].to_numpy(), gp["glicko"].to_numpy())

        # TEST 2 exact: same board (piece placement + side to move), different players.
        ee = e.assign(pos=e["fen_before"].str.split(" ", n=2).str[:2].str.join(" "))
        multi = ee.groupby("pos")["player_id"].nunique()
        multi_pos = set(multi[multi >= 2].index)
        per = (ee[ee["pos"].isin(multi_pos)].groupby(["pos", "player_id"])[up_col].max())
        varying = int(per.groupby("pos").nunique().ge(2).sum())
        t2 = val.within_position_skill(ee["pos"].to_numpy(), e["glicko"].to_numpy(),
                                       e[up_col].to_numpy(), n_boot=1000, seed=SEED)
        t2_exact = {"n_positions_multi": len(multi_pos), "n_positions_varying": varying}
        # TEST 2 substitute (feasible): move-level rating effect holding difficulty and
        # clock fixed. Not the clean same-position test; read the effect size, n is huge.
        t2_sub = val.partial_rank_correlation(
            e[up_col].to_numpy(), e["glicko"].to_numpy(),
            [e["maia_entropy"].to_numpy(), np.log1p(e["clock_before_s"].clip(lower=0).to_numpy()),
             (e["regime"] == "rapid").to_numpy(), (e["game_phase"] == "middlegame").to_numpy(),
             (e["game_phase"] == "endgame").to_numpy()])

        t3_pool = players.dropna(subset=["apr_up", "apr_wpl", "may_glicko"])
        t3_pool = t3_pool[(t3_pool["apr_n"] >= MIN_MOVES_PER_PLAYER) & (t3_pool["may_n"] >= 10)]
        t3 = val.predictive_validity(t3_pool["may_glicko"].to_numpy(),
                                     t3_pool["apr_up"].to_numpy(), t3_pool["apr_wpl"].to_numpy())

        out[opt] = {"n_events": n_events, "n_players_stable": len(stable),
                    "test1": t1, "test1_by_regime": t1_by, "test2": t2,
                    "test2_exact": t2_exact, "test2_stratified": t2_sub,
                    "test3": t3, "test3_n": len(t3_pool)}
    return out


def _detectable(d: dict, key: str) -> bool:
    """Directionally right and statistically detectable."""
    if key == "test1":
        return d["rho"] > 0 and d["lo"] > 0
    if key == "test2":
        return d["c"] > 0.5 and d["lo"] > 0.5
    if key == "test3":
        return d["partial_rho"] > 0 and d["p"] < 0.05
    return False


def _recommend(cutoffs: dict) -> dict:
    """TEST 2 exact is infeasible here (positions do not recur across players), so
    the decision rests on the feasible tests: rating monotonicity and predictive
    validity. 60s primary if both are detectable, else relax to 120s primary."""
    def passes(cut):
        b = cutoffs[str(cut)]["B"]
        return int(_detectable(b["test1"], "test1")) + int(_detectable(b["test3"], "test3"))
    p60, p120 = passes(60), passes(120)
    if p60 >= 2:
        rec = "keep 60s strict as primary (both feasible tests detectable)"
    elif p120 >= 2:
        rec = "relax to 120s as primary, keep 60s as the strict robustness reading"
    else:
        rec = "neither cutoff reaches clear significance on the feasible tests; underpowered"
    return {"feasible_tests_detectable_60": p60, "feasible_tests_detectable_120": p120,
            "recommendation": rec, "note": "TEST 2 exact infeasible (positions unique); "
            "recommendation uses TEST 1 and TEST 3"}


def _print_report(report: dict) -> None:
    log(f"\n=== STEP 5 skill validation (rating-free; Maia @ Elo {report['fixed_elo']}) ===")
    log(f"moves {report['n_moves']:,}, eligible {report['n_eligible']:,}")
    for cut, cd in report["cutoffs"].items():
        log(f"\n--- clock cutoff {cut}s ---")
        for opt in ["B", "A"]:
            d = cd[opt]
            tag = "PRIMARY" if opt == "B" else "robustness"
            log(f"  Option {opt} ({tag}): {d['n_events']:,} upgrade events, "
                f"{d['n_players_stable']:,} stable players")
            t1 = d["test1"]
            log(f"    TEST 1 rating monotonicity: rho {t1['rho']:+.3f} "
                f"CI[{t1['lo']:+.3f},{t1['hi']:+.3f}] p {t1['p']:.1e} n {t1['n']}  "
                f"{'DETECTABLE' if _detectable(t1,'test1') else 'weak'}")
            log("      by regime: " + ", ".join(
                f"{r}: rho {x['rho']:+.3f} (n {x['n']})" for r, x in d["test1_by_regime"].items()))
            ex = d["test2_exact"]
            sub = d["test2_stratified"]
            log(f"    TEST 2 same-position (decisive): INFEASIBLE, positions do not recur "
                f"across players ({ex['n_positions_multi']} boards with >=2 players, "
                f"{ex['n_positions_varying']} with upgrade variation)")
            log(f"      substitute (difficulty+clock controlled move-level rating effect): "
                f"partial rho {sub['rho']:+.3f} p {sub['p']:.1e} n {sub['n']:,} (small; read effect size)")
            t3 = d["test3"]
            log(f"    TEST 3 predictive: partial rho {t3['partial_rho']:+.3f} p {t3['p']:.1e} "
                f"(raw {t3['raw_rho']:+.3f}) n {d['test3_n']}  "
                f"{'DETECTABLE' if _detectable(t3,'test3') else 'weak'}")
    r = report["clock_recommendation"]
    log("\n=== Clock cutoff decision (TEST 2 exact infeasible; using TEST 1 + TEST 3) ===")
    log(f"  feasible tests detectable at 60s: {r['feasible_tests_detectable_60']}/2, "
        f"at 120s: {r['feasible_tests_detectable_120']}/2")
    log(f"  recommendation: {r['recommendation']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "engine", "classify"])
    args = ap.parse_args()
    cfg = load_config()
    {"sample": stage_sample, "engine": stage_engine, "classify": stage_classify}[args.stage](cfg)


if __name__ == "__main__":
    main()
