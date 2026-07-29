"""Maia complexity validation, STEP B (sample and analyze stages).

Runs in the MAIN x86_64 venv. The Maia inference itself runs separately in the
arm64 maia venv (scripts/maia_infer.py) because torch there needs numpy<2. The
handoff is on disk: `sample` writes the FEN list and think-time, `maia_infer`
writes the dispersion, `analyze` joins them and runs the STEP 1 validation.

    python scripts/maia_validation.py sample     # 20k FENs + think-time + scoring tasks
    .venv_maia/bin/python scripts/maia_infer.py   # Maia dispersion (arm64 venv, 90 min ceiling)
    python scripts/maia_validation.py analyze     # 3 checks vs think-time, entropy-vs-Maia, verdict

Nothing here touches the existing raw-evals cache or dataset. Everything lands
under data/processed/maia_val/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chess_strength import maia
from chess_strength import thinktime as tt
from chess_strength.config import load_config

# Same sampling knobs as the resolution pilot, so this is apples-to-apples.
VAL_N = 20000
SATURATED_FRACTION = 0.80
SATURATED_THRESHOLD = 0.95
SEED = 0
VAL_DIR_NAME = "maia_val"


def log(msg: str) -> None:
    print(msg, flush=True)


def _paths(cfg: dict):
    proc = Path(cfg["paths"]["processed"])
    return proc, proc / VAL_DIR_NAME


def _sn_curve(sub: pd.DataFrame, comp_col: str, nbins: int = 10):
    """Decile means of think-time along a complexity column, Pearson, and whether
    the top decile turns over (drops below the peak by >5%). Mirrors the pilot."""
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


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def stage_sample(cfg: dict) -> None:
    proc, val = _paths(cfg)
    val.mkdir(parents=True, exist_ok=True)

    comp = tt.load_complexity(proc / "tierb" / "cache")
    sat = comp[comp["complexity"] >= SATURATED_THRESHOLD]
    rest = comp[comp["complexity"] < SATURATED_THRESHOLD]
    n_sat = round(VAL_N * SATURATED_FRACTION)
    n_rest = VAL_N - n_sat
    log(f"pool: saturated {len(sat):,}, below {len(rest):,}")
    val_fens = pd.concat([
        sat.sample(n=min(n_sat, len(sat)), random_state=SEED),
        rest.sample(n=min(n_rest, len(rest)), random_state=SEED + 1),
    ]).reset_index(drop=True).rename(columns={"complexity": "old_complexity"})
    val_fens["saturated"] = val_fens["old_complexity"] >= SATURATED_THRESHOLD
    val_fens.to_parquet(val / "val_fens.parquet", index=False)
    log(f"sampled {len(val_fens):,} FENs: {int(val_fens['saturated'].sum()):,} saturated, "
        f"{int((~val_fens['saturated']).sum()):,} below (seed {SEED})")

    # Think-time and confounds for those positions (same assembly as STEP 1).
    df = tt.assemble(cfg, proc)
    keep = df[df["fen_before"].isin(set(val_fens["fen"]))].copy()
    keep = keep.rename(columns={"complexity": "old_complexity"})
    keep["band_elo"] = keep["rating_band"].map(maia.band_to_elo)
    cols = ["fen_before", "time_spent_s", "log_clock_before", "move_number",
            "game_phase", "regime", "rating_band", "band_elo", "old_complexity"]
    keep[cols].to_parquet(val / "val_moves.parquet", index=False)

    # One scoring task per unique (position, regime, band Elo). Most FENs occur
    # once, so this is close to the FEN count. Maia is scored on these.
    tasks = keep[["fen_before", "regime", "band_elo"]].drop_duplicates()
    tasks = tasks.rename(columns={"fen_before": "fen"})
    tasks.to_parquet(val / "val_tasks.parquet", index=False)
    log(f"think-time rows {len(keep):,} over {keep['fen_before'].nunique():,} FENs; "
        f"scoring tasks {len(tasks):,} (regime x band)")
    log(f"regime split of tasks: {tasks['regime'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _load_dispersion(val: Path) -> pd.DataFrame:
    parts = sorted((val / "dispersion").glob("batch_*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def _checks(m: pd.DataFrame, comp_col: str, label: str, report: dict) -> None:
    """The three STEP 1 style checks for one complexity column."""
    raw_rho, _, raw_n = tt.spearman(m[comp_col].to_numpy(dtype=float), m["time_spent_s"].to_numpy())
    _, pear, turn, top, peak = _sn_curve(m, comp_col)
    sat = m[m["old_complexity"] >= SATURATED_THRESHOLD]
    s_rho, _, s_n = tt.spearman(sat[comp_col].to_numpy(dtype=float), sat["time_spent_s"].to_numpy())
    _, s_pear, s_turn, _, _ = _sn_curve(sat, comp_col)
    report[label] = {
        "raw_spearman": raw_rho, "raw_n": int(raw_n),
        "binned_pearson": pear, "turnover": turn,
        "top_decile_mean_s": top, "peak_decile_mean_s": peak,
        "within_saturated": {"raw_spearman": s_rho, "binned_pearson": s_pear,
                             "turnover": s_turn, "n": int(s_n)},
    }
    log(f"  {label:16s} raw Spearman {raw_rho:+.3f} (n {raw_n:,}), binned Pearson {pear:+.3f}, "
        f"turnover={turn}")
    log(f"  {'':16s}   within old-saturated: raw {s_rho:+.3f}, binned Pearson {s_pear:+.3f}, "
        f"turnover={s_turn}, n {s_n:,}")


def stage_analyze(cfg: dict) -> None:
    _, val = _paths(cfg)
    moves = pd.read_parquet(val / "val_moves.parquet")
    disp = _load_dispersion(val)
    tasks = pd.read_parquet(val / "val_tasks.parquet")
    if len(disp) < len(tasks):
        log(f"dispersion incomplete ({len(disp):,}/{len(tasks):,} tasks). Finish maia_infer first.")
        raise SystemExit(1)

    m = moves.merge(disp, left_on=["fen_before", "regime", "band_elo"],
                    right_on=["fen", "regime", "band_elo"], how="inner")
    log(f"joined dispersion onto {len(m):,} move rows")
    report: dict = {"n_rows": len(m), "n_fens": int(m["fen_before"].nunique()),
                    "config": {"VAL_N": VAL_N, "SATURATED_FRACTION": SATURATED_FRACTION,
                               "SATURATED_THRESHOLD": SATURATED_THRESHOLD, "SEED": SEED}}

    # Check 1 + 2 together: run the S(n) / turnover / saturated-zone checks for
    # engine entropy (old) and each Maia reading, side by side.
    log("\n=== Checks 1 and 2: S(n) tracking, turnover, and the saturated zone ===")
    report["checks"] = {}
    _checks(m, "old_complexity", "engine_entropy", report["checks"])
    for col in ["maia_entropy", "maia_entropy_norm", "maia_top1"]:
        _checks(m, col, col, report["checks"])

    # Check 3: raw Spearman by rating band, primary metric.
    log("\n=== Check 3: raw Spearman(maia_entropy, think-time) by rating band ===")
    report["by_band"] = {}
    for band, g in m.groupby("rating_band"):
        rho, _, n = tt.spearman(g["maia_entropy"].to_numpy(), g["time_spent_s"].to_numpy())
        report["by_band"][str(band)] = {"rho": rho, "n": int(n)}
        log(f"  {band!s:16s} rho={rho:+.3f}  n={n:,}")
    log("  by regime (maia_entropy):")
    report["by_regime"] = {}
    for reg, g in m.groupby("regime"):
        rho, _, n = tt.spearman(g["maia_entropy"].to_numpy(), g["time_spent_s"].to_numpy())
        _, pe, turn, _, _ = _sn_curve(g, "maia_entropy")
        report["by_regime"][reg] = {"rho": rho, "binned_pearson": pe, "turnover": turn, "n": int(n)}
        log(f"    {reg:6s} rho={rho:+.3f}  binned Pearson={pe:+.3f}  turnover={turn}  n={n:,}")

    # Verdict: primary metric is maia_entropy.
    prim = report["checks"]["maia_entropy"]
    eng = report["checks"]["engine_entropy"]
    no_turnover = not prim["turnover"]
    tracks = prim["binned_pearson"] > 0.3 and prim["raw_spearman"] > 0
    sat_signal = prim["within_saturated"]["binned_pearson"] > 0.2 and prim["within_saturated"]["raw_spearman"] > 0
    go = bool(no_turnover and tracks and sat_signal)
    report["verdict"] = {
        "go": go, "no_turnover": bool(no_turnover), "tracks_thinktime": bool(tracks),
        "signal_in_saturated_zone": bool(sat_signal),
        "engine_entropy_saturated_zone_pearson": eng["within_saturated"]["binned_pearson"],
        "maia_entropy_saturated_zone_pearson": prim["within_saturated"]["binned_pearson"],
    }
    log("\n=== GO / NO-GO for Maia as the primary complexity gate ===")
    log(f"  Maia tracks think-time ............. {tracks}  (binned Pearson {prim['binned_pearson']:+.3f}, raw {prim['raw_spearman']:+.3f})")
    log(f"  no top-end turnover ............... {no_turnover}  (engine entropy turnover was {eng['turnover']})")
    log(f"  signal in old-saturated zone ...... {sat_signal}  (Maia {prim['within_saturated']['binned_pearson']:+.3f} vs engine {eng['within_saturated']['binned_pearson']:+.3f})")
    log(f"  VERDICT ........................... {'GO' if go else 'NO-GO'}")

    (val / "maia_report.json").write_text(json.dumps(report, indent=2, default=str))
    _plot(val, m)
    log(f"\nsaved report: {val / 'maia_report.json'}")


def _plot(val: Path, m: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sat = m[m["old_complexity"] >= SATURATED_THRESHOLD]
    ax[0].hist(sat["old_complexity"], bins=40, alpha=0.55, label="engine entropy (old)", color="#c05621")
    ax[0].hist(sat["maia_entropy_norm"], bins=40, alpha=0.55, label="Maia dispersion (norm)", color="#2b6cb0")
    ax[0].set_title("Old-saturated positions:\ndoes Maia dispersion spread them out?")
    ax[0].set_xlabel("normalized entropy")
    ax[0].set_ylabel("positions")
    ax[0].legend(fontsize=8)
    for col, color, lab in [("old_complexity", "#c05621", "engine entropy"),
                            ("maia_entropy", "#2b6cb0", "Maia dispersion")]:
        g, _, _, _, _ = _sn_curve(m, col)
        ax[1].plot(g["bin"], g["mean_t"], "o-", color=color, label=lab)
    ax[1].set_title("S(n): mean think-time by complexity decile\n(does the top turn over?)")
    ax[1].set_xlabel("complexity decile (0 = lowest)")
    ax[1].set_ylabel("mean think-time (s)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(val / "maia_validation.png", dpi=130)
    log(f"saved figure: {val / 'maia_validation.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "analyze"])
    args = ap.parse_args()
    cfg = load_config()
    {"sample": stage_sample, "analyze": stage_analyze}[args.stage](cfg)


if __name__ == "__main__":
    main()
