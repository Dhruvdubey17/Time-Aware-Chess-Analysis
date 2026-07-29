"""PART 0: is Maia dispersion still a think-time signal after surface controls?

The worry: Maia dispersion had a strong raw correlation with think-time, but maybe
it was just re-encoding how many legal moves there are (more moves to look at
takes longer) or other surface features. This regresses think-time on Maia
dispersion while holding number of legal moves, material, phase, move number,
clock remaining, regime, and rating band. If the dispersion coefficient stays
clearly positive and significant, it is a real difficulty signal. If it collapses
once legal-move count is in, it was mostly move-count and we should stop.

Runs on the 20k positions already scored in the last run. No engine, no Maia,
no new data.
"""

from __future__ import annotations

import glob

import chess
import numpy as np
import pandas as pd

from chess_strength import thinktime as tt
from chess_strength.config import load_config
from chess_strength.features import material_cp

NUMERIC = ["maia_entropy", "n_legal", "abs_material", "move_number", "log_clock_before"]
CATEGORICAL = ["game_phase", "regime", "rating_band"]


def log(msg: str) -> None:
    print(msg, flush=True)


def _design(df: pd.DataFrame, numeric: list[str]) -> tuple[np.ndarray, list[str]]:
    """Intercept, z-scored numerics, and drop-first dummies for the categoricals."""
    cols = [np.ones(len(df))]
    names = ["intercept"]
    for c in numeric:
        v = df[c].to_numpy(dtype=float)
        sd = v.std()
        cols.append((v - v.mean()) / sd if sd > 0 else v * 0.0)
        names.append(c)
    for c in CATEGORICAL:
        d = pd.get_dummies(df[c], prefix=c, drop_first=True)
        for name in d.columns:
            cols.append(d[name].to_numpy(dtype=float))
            names.append(name)
    return np.column_stack(cols), names


def _maia_coef(df: pd.DataFrame, numeric: list[str]) -> dict:
    y = np.log1p(df["time_spent_s"].to_numpy(dtype=float))
    X, names = _design(df, numeric)
    fit = tt.ols_fit(X, y)
    i = names.index("maia_entropy")
    return {"beta": float(fit["beta"][i]), "t": float(fit["t"][i]),
            "p": float(fit["p"][i]), "r2": fit["r2"], "n": fit["n"]}


def main() -> None:
    cfg = load_config()
    vdir = f"{cfg['paths']['processed']}/maia_val"

    moves = pd.read_parquet(f"{vdir}/val_moves.parquet")
    disp = pd.concat(
        (pd.read_parquet(p) for p in sorted(glob.glob(f"{vdir}/dispersion/batch_*.parquet"))),
        ignore_index=True,
    )
    m = moves.merge(disp[["fen", "regime", "band_elo", "maia_entropy"]],
                    left_on=["fen_before", "regime", "band_elo"],
                    right_on=["fen", "regime", "band_elo"], how="inner")
    log(f"rows: {len(m):,}")

    # Surface features from the FEN: legal-move count and material imbalance.
    n_legal, abs_mat = [], []
    for fen in m["fen_before"]:
        b = chess.Board(fen)
        n_legal.append(b.legal_moves.count())
        abs_mat.append(abs(material_cp(fen, fen.split()[1] == "w")))
    m["n_legal"] = n_legal
    m["abs_material"] = abs_mat

    log(f"\nmaia_entropy vs n_legal correlation (Pearson): "
        f"{np.corrcoef(m['maia_entropy'], m['n_legal'])[0, 1]:+.3f}")

    # Standardized coefficient of maia_entropy on log think-time, three nested models.
    raw = _maia_coef(m, ["maia_entropy"])
    no_nlegal = _maia_coef(m, ["maia_entropy", "abs_material", "move_number", "log_clock_before"])
    full = _maia_coef(m, NUMERIC)

    log("\n=== Standardized maia_entropy coefficient on log(think-time) ===")
    log(f"  raw (no controls)          beta={raw['beta']:+.4f}  t={raw['t']:+.1f}  p={raw['p']:.1e}  R2={raw['r2']:.3f}")
    log(f"  + controls, no n_legal     beta={no_nlegal['beta']:+.4f}  t={no_nlegal['t']:+.1f}  p={no_nlegal['p']:.1e}  R2={no_nlegal['r2']:.3f}")
    log(f"  + all controls incl n_legal beta={full['beta']:+.4f}  t={full['t']:+.1f}  p={full['p']:.1e}  R2={full['r2']:.3f}")

    kept = full["beta"] / raw["beta"] if raw["beta"] else 0.0
    survives = full["beta"] > 0 and full["p"] < 0.01
    log(f"\n  fraction of raw effect kept under full controls: {kept:.0%}")
    log(f"  VERDICT: Maia dispersion {'SURVIVES' if survives else 'COLLAPSES'} the controls "
        f"(coef {'positive and significant' if survives else 'gone'}).")
    if not survives:
        log("  -> STOP and report: dispersion was mostly surface features.")
    else:
        log("  -> proceed to build the classifier.")


if __name__ == "__main__":
    main()
