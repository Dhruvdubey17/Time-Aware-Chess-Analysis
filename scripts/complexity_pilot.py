"""Resolution pilot: does deeper engine annotation fix the complexity saturation?

STEP 1 found the top-4 decision-entropy metric pins near 1.0 (median 0.975, 61%
of positions >= 0.95) and the S(n) think-time curve turns over at the top because
the saturated bin mixes truly hard positions with dull equal ones. Hypothesis:
more nodes and higher multipv will SEPARATE hard from easy at the top instead of
saturating, and the separated metric will track think-time without a turnover.

This is a small PILOT before any full re-annotation. Three stages, run in order:

    python scripts/complexity_pilot.py sample     # pick FENs, build think-time, warmup timing
    python scripts/complexity_pilot.py annotate   # real engine work, 6 workers, 60 min ceiling
    python scripts/complexity_pilot.py analyze     # recompute complexity, run the 3 checks, verdict

It only reads the existing cache (never writes it) and writes everything under a
separate pilot dir. Fixed node budget, single-threaded engine per worker, per the
standing rules.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from chess_strength import thinktime as tt
from chess_strength.config import load_config

# ---- pilot config (easy to change) ----------------------------------------
PILOT_N = 8000              # positions to re-annotate
SATURATED_FRACTION = 0.80   # share drawn from the currently-saturated zone
SATURATED_THRESHOLD = 0.95  # current entropy >= this counts as saturated
HI_NODES = 300000           # 10x the existing 30000-node budget
HI_MULTIPV = 10             # existing cache used multipv 4
WORKERS = 6                 # M3 Air, single-threaded engine per worker
SEED = 0
CEILING_MIN = 60            # hard wall-clock safety stop for annotation
BATCH = 100                 # FENs per work unit and per atomic flush
HASH_MB = 64                # per-engine hash, fixed for reproducibility
MATE_CP = 10000             # mate stored as a large signed cp, matches the cache
SERIAL_BASELINE_PER_MIN = 1475.0  # the old 30k-node serial rate, for context

PILOT_DIR_NAME = "tierb_pilot"


def log(msg: str) -> None:
    print(msg, flush=True)


def _paths(cfg: dict):
    proc = Path(cfg["paths"]["processed"])
    pilot = proc / PILOT_DIR_NAME
    return proc, pilot, pilot / "cache"


# ---------------------------------------------------------------------------
# Worker side: one persistent Stockfish per process, killed by pool.terminate
# ---------------------------------------------------------------------------
_ENGINE = None
_NODES = None
_MULTIPV = None


def _init_worker(sf_path: str, nodes: int, multipv: int, hash_mb: int) -> None:
    global _ENGINE, _NODES, _MULTIPV
    import chess.engine

    _ENGINE = chess.engine.SimpleEngine.popen_uci(sf_path)
    _ENGINE.configure({"Threads": 1, "Hash": hash_mb})
    _NODES = nodes
    _MULTIPV = multipv


def _annotate_batch(fens: list[str]) -> list[dict]:
    import chess
    from chess.engine import Limit

    out = []
    for fen in fens:
        board = chess.Board(fen)
        infos = _ENGINE.analyse(board, Limit(nodes=_NODES), multipv=_MULTIPV)
        cps = sorted(
            (i["score"].relative.score(mate_score=MATE_CP) for i in infos), reverse=True
        )
        out.append({"fen": fen, "cps": cps, "n_legal": board.legal_moves.count()})
    return out


def _flush_batch(rows: list[dict], cache_dir: Path, idx: int) -> None:
    """Write one batch atomically: temp file then rename, so a stop loses nothing."""
    df = pd.DataFrame(rows)
    df["cps"] = df["cps"].apply(list)
    tmp = cache_dir / f".batch_{idx:05d}.tmp.parquet"
    final = cache_dir / f"batch_{idx:05d}.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)


# ---------------------------------------------------------------------------
# STEP A: sample positions and pull their human think-time
# ---------------------------------------------------------------------------

def stage_sample(cfg: dict) -> None:
    proc, pilot, _ = _paths(cfg)
    tierb = proc / "tierb"
    pilot.mkdir(parents=True, exist_ok=True)

    # Current complexity for every cached FEN (read-only on the existing cache).
    comp = tt.load_complexity(tierb / "cache")
    sat = comp[comp["complexity"] >= SATURATED_THRESHOLD]
    rest = comp[comp["complexity"] < SATURATED_THRESHOLD]
    n_sat = round(PILOT_N * SATURATED_FRACTION)
    n_rest = PILOT_N - n_sat
    log(f"pool: saturated (>= {SATURATED_THRESHOLD}) {len(sat):,}, below {len(rest):,}")

    sat_s = sat.sample(n=min(n_sat, len(sat)), random_state=SEED)
    rest_s = rest.sample(n=min(n_rest, len(rest)), random_state=SEED + 1)
    pilot_fens = pd.concat([sat_s, rest_s]).reset_index(drop=True)
    pilot_fens = pilot_fens.rename(
        columns={"complexity": "old_complexity", "eval_gap_1_2": "old_eval_gap",
                 "n_reasonable": "old_n_reasonable"}
    )
    pilot_fens["saturated"] = pilot_fens["old_complexity"] >= SATURATED_THRESHOLD
    pilot_fens.to_parquet(pilot / "pilot_fens.parquet", index=False)
    log(f"sampled {len(pilot_fens):,} FENs: {int(pilot_fens['saturated'].sum()):,} saturated, "
        f"{int((~pilot_fens['saturated']).sum()):,} below (seed {SEED})")

    # Think-time and confounds for those positions, reusing the STEP 1 assembly.
    df = tt.assemble(cfg, proc)
    keep = df[df["fen_before"].isin(set(pilot_fens["fen"]))].copy()
    keep = keep.rename(columns={"complexity": "old_complexity"})
    cols = ["fen_before", "time_spent_s", "log_time_spent", "log_clock_before",
            "move_number", "game_phase", "regime", "rating_band", "old_complexity"]
    keep[cols].to_parquet(pilot / "pilot_moves.parquet", index=False)
    n_fen_cov = keep["fen_before"].nunique()
    log(f"pilot think-time rows: {len(keep):,} over {n_fen_cov:,} of {len(pilot_fens):,} FENs "
        f"({n_fen_cov / len(pilot_fens):.1%} have at least one observed move)")

    # Serial warmup at the pilot settings, to size the run and the speedup.
    _warmup(cfg, pilot_fens["fen"].tolist(), n=25)


def _warmup(cfg: dict, fens: list[str], n: int) -> None:
    import chess
    import chess.engine
    from chess.engine import Limit

    n = min(n, len(fens))
    log(f"\nserial warmup: {n} FENs at {HI_NODES} nodes, multipv {HI_MULTIPV} ...")
    start = time.time()
    with chess.engine.SimpleEngine.popen_uci(cfg["stockfish_path"]) as eng:
        eng.configure({"Threads": 1, "Hash": HASH_MB})
        for fen in fens[:n]:
            eng.analyse(chess.Board(fen), Limit(nodes=HI_NODES), multipv=HI_MULTIPV)
    per = (time.time() - start) / n
    serial_rate = 60.0 / per
    log(f"  serial: {per:.3f}s/pos, {serial_rate:.0f} FENs/min (one worker)")
    # Rough projection at WORKERS. The Tier-B run saw ~3.1x on this machine.
    for sx in (3.0, 3.5):
        proj_min = PILOT_N / (serial_rate * sx) if serial_rate > 0 else float("inf")
        log(f"  projected {PILOT_N} FENs at {WORKERS} workers, {sx:.1f}x speedup: {proj_min:.0f} min")
    log(f"  ceiling is {CEILING_MIN} min. (old 30k-node serial baseline was "
        f"~{SERIAL_BASELINE_PER_MIN:.0f}/min, expected ~10x slower here.)")


# ---------------------------------------------------------------------------
# STEP B: annotate the pilot at high resolution, in parallel, with a ceiling
# ---------------------------------------------------------------------------

def _done_fens(cache_dir: Path) -> set[str]:
    done = set()
    for b in sorted(cache_dir.glob("batch_*.parquet")):
        done |= set(pd.read_parquet(b, columns=["fen"])["fen"])
    return done


def stage_annotate(cfg: dict) -> None:
    proc, pilot, cache_dir = _paths(cfg)
    assert cache_dir != (proc / "tierb" / "cache"), "pilot must not write the existing cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pilot_fens = pd.read_parquet(pilot / "pilot_fens.parquet")
    already = _done_fens(cache_dir)
    todo = [f for f in pilot_fens["fen"].tolist() if f not in already]
    start_idx = len(list(cache_dir.glob("batch_*.parquet")))
    log(f"pilot FENs {len(pilot_fens):,}, already done {len(already):,}, to do {len(todo):,}")
    if not todo:
        log("nothing to annotate, cache already complete.")
        return

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    ceiling_s = CEILING_MIN * 60
    start = time.time()
    done = 0
    hit_ceiling = False

    pool = mp.Pool(
        WORKERS, initializer=_init_worker,
        initargs=(cfg["stockfish_path"], HI_NODES, HI_MULTIPV, HASH_MB),
    )
    try:
        for k, rows in enumerate(pool.imap_unordered(_annotate_batch, batches)):
            _flush_batch(rows, cache_dir, start_idx + k)
            done += len(rows)
            elapsed = time.time() - start
            rate = done / elapsed * 60 if elapsed > 0 else 0.0
            log(f"  flushed batch {start_idx + k:05d}  done {done:,}/{len(todo):,}  "
                f"{elapsed / 60:.1f} min  {rate:.0f} FENs/min")
            if elapsed > ceiling_s:
                hit_ceiling = True
                log(f"  CEILING {CEILING_MIN} min hit, stopping.")
                break
    finally:
        # Hard kill: workers die, Stockfish sees stdin EOF and exits. No blocking
        # quit(), which is what hung under Rosetta before.
        pool.terminate()
        pool.join()

    elapsed = time.time() - start
    rate = done / elapsed * 60 if elapsed > 0 else 0.0
    total_done = len(_done_fens(cache_dir))
    log(f"\nannotated {done:,} this run in {elapsed / 60:.1f} min, {rate:.0f} FENs/min at "
        f"{WORKERS} workers.")
    log(f"pilot cache now holds {total_done:,}/{len(pilot_fens):,} FENs.")
    if hit_ceiling or total_done < len(pilot_fens):
        log(f"INCOMPLETE ({total_done:,}/{len(pilot_fens):,}). Not running analysis on a "
            f"partial, order-biased sample. Re-run 'annotate' to resume, or lower PILOT_N.")


# ---------------------------------------------------------------------------
# STEP C: recompute complexity and re-run the STEP 1 validation on the pilot
# ---------------------------------------------------------------------------

def _sn_curve(sub: pd.DataFrame, comp_col: str, nbins: int = 10):
    """Decile means of think-time along a complexity column, plus Pearson and
    whether the top decile turns over (drops below the peak by >5%)."""
    s = sub.dropna(subset=[comp_col, "time_spent_s"]).copy()
    s["bin"] = pd.qcut(s[comp_col].rank(method="first"), nbins, labels=False, duplicates="drop")
    g = s.groupby("bin").agg(
        n=(comp_col, "size"), mean_c=(comp_col, "mean"),
        mean_t=("time_spent_s", "mean"), median_t=("time_spent_s", "median"),
    ).reset_index()
    pear = float(np.corrcoef(g["mean_c"], g["mean_t"])[0, 1]) if len(g) > 2 else float("nan")
    peak, top = float(g["mean_t"].max()), float(g["mean_t"].iloc[-1])
    turnover = bool(top < 0.95 * peak)
    return g, pear, turnover, top, peak


def stage_analyze(cfg: dict) -> None:
    import json

    _, pilot, cache_dir = _paths(cfg)
    pilot_fens = pd.read_parquet(pilot / "pilot_fens.parquet")
    pm = pd.read_parquet(pilot / "pilot_moves.parquet")

    done = _done_fens(cache_dir)
    if len(done) < len(pilot_fens):
        log(f"pilot cache incomplete ({len(done):,}/{len(pilot_fens):,}). Finish 'annotate' first.")
        sys.exit(1)

    # New (high-resolution) complexity per FEN.
    new = tt.load_complexity(cache_dir).rename(
        columns={"complexity": "new_complexity", "eval_gap_1_2": "new_eval_gap",
                 "n_reasonable": "new_n_reasonable"}
    )
    fens = pilot_fens.merge(new, on="fen", how="inner")
    sat = fens[fens["saturated"]]
    report: dict = {"config": {
        "PILOT_N": PILOT_N, "SATURATED_FRACTION": SATURATED_FRACTION,
        "SATURATED_THRESHOLD": SATURATED_THRESHOLD, "HI_NODES": HI_NODES,
        "HI_MULTIPV": HI_MULTIPV, "WORKERS": WORKERS, "SEED": SEED,
    }}

    # --- Check 1: do the old-saturated positions spread out now? ---
    log("\n=== Check 1: do previously-saturated positions (old entropy >= 0.95) spread out? ===")
    q = sat["new_complexity"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3)
    still = float((sat["new_complexity"] >= SATURATED_THRESHOLD).mean())
    spread = {
        "n_saturated": len(sat),
        "old_median": float(sat["old_complexity"].median()),
        "old_std": float(sat["old_complexity"].std()),
        "new_median": float(sat["new_complexity"].median()),
        "new_std": float(sat["new_complexity"].std()),
        "new_share_still_ge_0.95": still,
        "new_deciles": {str(k): float(v) for k, v in q.items()},
    }
    report["check1_spread"] = spread
    log(f"  saturated group n={spread['n_saturated']:,}")
    log(f"  OLD: median {spread['old_median']:.3f}, std {spread['old_std']:.3f} (pinned at ceiling)")
    log(f"  NEW: median {spread['new_median']:.3f}, std {spread['new_std']:.3f}")
    log(f"  NEW share still >= {SATURATED_THRESHOLD}: {still:.1%}")
    log(f"  NEW quantiles (10/25/50/75/90): {list(q)}")

    # --- Check 3: old vs new on the same positions ---
    log("\n=== Check 3: old vs new complexity on the same positions ===")
    pear = float(np.corrcoef(fens["old_complexity"], fens["new_complexity"])[0, 1])
    rho, _, _ = tt.spearman(fens["old_complexity"].to_numpy(), fens["new_complexity"].to_numpy())
    report["check3_old_vs_new"] = {"pearson": pear, "spearman": rho, "n": len(fens)}
    log(f"  old vs new: Pearson {pear:+.3f}, Spearman {rho:+.3f}, n {len(fens):,}")
    log(f"  the saturated group had old std {spread['old_std']:.3f} (no spread) and now has "
        f"new std {spread['new_std']:.3f}: high resolution {'separates' if spread['new_std'] > 0.10 else 'does NOT separate'} them.")

    # --- Check 2: does new complexity track think-time without a turnover? ---
    log("\n=== Check 2: high-resolution S(n) vs think-time (turnover?) ===")
    m = pm.merge(new[["fen", "new_complexity"]], left_on="fen_before", right_on="fen", how="inner")
    report["check2_sn"] = {}
    curves = {}
    for label, col in [("OLD", "old_complexity"), ("NEW", "new_complexity")]:
        raw_rho, _, raw_n = tt.spearman(m[col].to_numpy(), m["time_spent_s"].to_numpy())
        g, pe, turn, top, peak = _sn_curve(m, col)
        curves[label] = g
        report["check2_sn"][label] = {
            "raw_spearman": raw_rho, "raw_n": int(raw_n), "binned_pearson": pe,
            "turnover": turn, "top_decile_mean_s": top, "peak_decile_mean_s": peak,
        }
        log(f"  {label}: raw Spearman {raw_rho:+.3f} (n {raw_n:,}), binned Pearson {pe:+.3f}, "
            f"turnover={turn} (top decile {top:.1f}s vs peak {peak:.1f}s)")
    # By regime, new only.
    log("  NEW by regime:")
    report["check2_sn"]["NEW_by_regime"] = {}
    for reg, gr in m.groupby("regime"):
        raw_rho, _, raw_n = tt.spearman(gr["new_complexity"].to_numpy(), gr["time_spent_s"].to_numpy())
        _, pe, turn, top, peak = _sn_curve(gr, "new_complexity")
        report["check2_sn"]["NEW_by_regime"][reg] = {
            "raw_spearman": raw_rho, "binned_pearson": pe, "turnover": turn, "n": int(raw_n)}
        log(f"    {reg:6s} raw {raw_rho:+.3f}, binned Pearson {pe:+.3f}, turnover={turn}, n {raw_n:,}")

    # Sharpest test: within the old-saturated positions, does new complexity track time?
    ms = m[m["fen_before"].isin(set(sat["fen"]))]
    raw_rho_s, _, n_s = tt.spearman(ms["new_complexity"].to_numpy(), ms["time_spent_s"].to_numpy())
    _, pe_s, turn_s, _, _ = _sn_curve(ms, "new_complexity")
    report["check2_sn"]["within_old_saturated"] = {
        "raw_spearman": raw_rho_s, "binned_pearson": pe_s, "turnover": turn_s, "n": int(n_s)}
    log(f"  within old-saturated only: NEW raw Spearman {raw_rho_s:+.3f}, binned Pearson "
        f"{pe_s:+.3f}, turnover={turn_s}, n {n_s:,}")

    # --- Verdict ---
    spread_ok = (spread["new_std"] > 0.10) and (still < 0.5) and (spread["new_median"] < SATURATED_THRESHOLD)
    no_turnover = not report["check2_sn"]["NEW"]["turnover"]
    tracks = report["check2_sn"]["NEW"]["binned_pearson"] > 0.3
    go = bool(spread_ok and no_turnover and tracks)
    report["verdict"] = {
        "go": go, "spread_ok": bool(spread_ok), "no_turnover": bool(no_turnover),
        "tracks_thinktime": bool(tracks),
    }
    log("\n=== GO / NO-GO for the full re-annotation ===")
    log(f"  saturated positions spread out ..... {spread_ok}  (new std {spread['new_std']:.3f}, "
        f"still>=0.95 {still:.1%}, new median {spread['new_median']:.3f})")
    log(f"  S(n) no top-end turnover ........... {no_turnover}")
    log(f"  new complexity tracks think-time ... {tracks}  (binned Pearson "
        f"{report['check2_sn']['NEW']['binned_pearson']:+.3f})")
    log(f"  VERDICT ............................ {'GO' if go else 'NO-GO'}")

    (pilot / "pilot_report.json").write_text(json.dumps(report, indent=2, default=str))
    _plot(pilot, sat, curves)
    log(f"\nsaved report: {pilot / 'pilot_report.json'}")


def _plot(pilot: Path, sat: pd.DataFrame, curves: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(sat["old_complexity"], bins=40, alpha=0.6, label="old (30k, mpv4)", color="#c05621")
    ax[0].hist(sat["new_complexity"], bins=40, alpha=0.6, label=f"new ({HI_NODES//1000}k, mpv{HI_MULTIPV})", color="#2b6cb0")
    ax[0].set_title("Old-saturated positions:\ndoes higher resolution spread them out?")
    ax[0].set_xlabel("decision entropy")
    ax[0].set_ylabel("positions")
    ax[0].legend(fontsize=8)
    for label, color in [("OLD", "#c05621"), ("NEW", "#2b6cb0")]:
        g = curves[label]
        ax[1].plot(g["bin"], g["mean_t"], "o-", color=color, label=f"{label} complexity")
    ax[1].set_title("S(n): mean think-time by complexity decile\n(does the top turn over?)")
    ax[1].set_xlabel("complexity decile (0 = lowest)")
    ax[1].set_ylabel("mean think-time (s)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pilot / "pilot_resolution.png", dpi=130)
    log(f"saved figure: {pilot / 'pilot_resolution.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "annotate", "analyze"])
    args = ap.parse_args()
    cfg = load_config()
    {"sample": stage_sample, "annotate": stage_annotate, "analyze": stage_analyze}[args.stage](cfg)


if __name__ == "__main__":
    main()
