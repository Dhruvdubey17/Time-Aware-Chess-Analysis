"""STEP 1: does position complexity predict how long a human thinks?

Option B (the primary upgrade rule) multiplies move quality by position
complexity, so complexity has to mean something real. The test here is simple:
in harder positions, do people actually spend more time on the clock? If yes,
Option B stands. If not, we fall back to Option A.

Complexity is read from the raw-evals cache, not recomputed. For each position
the cache holds `cps`, the side-to-move centipawn scores of Stockfish's top few
moves (best first, fixed node budget, multipv 4), and `n_legal`. The PRIMARY
complexity metric is the decision entropy of those scores: near 0 when one move
is clearly best, near 1 when several look equally good. A high-entropy position
hands the player a real choice to work out, so we expect it to cost more time.

Two robustness readings come off the same cache row: `eval_gap_1_2` (best minus
second best, Sunde's distinctiveness, which runs the other way because a clear
best move is easy) and `n_reasonable` (how many moves sit within a pawn of best).

We also fit and save an expected-think-time mapping. Later steps use it to
(a) scale the complexity term and (b) define genuine time pressure as the
residual: time actually spent minus what the position and clock context predict.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .complexity import complexity_features
from .stream_filter import classify_time_control

# Columns pulled from the interim per-move table to build think-time.
INTERIM_COLS = ["game_id", "player_id", "color", "ply", "move_number", "time_control", "clock_s"]
# A move faster than this is a premove, not real thinking, so we drop it.
MIN_THINK_S = 0.3

# The saved mapping predicts log(1 + seconds) from these columns. Later steps
# must build a frame with the same column names to score positions.
NUMERIC_FEATURES = ["complexity", "move_number", "log_clock_before"]
CATEGORICAL_FEATURES = ["game_phase", "regime", "rating_band"]
TARGET = "log_time_spent"


# ---------------------------------------------------------------------------
# Complexity from a cached row
# ---------------------------------------------------------------------------

def position_complexity(cps: list[int], n_legal: int) -> dict:
    """Complexity readings for one cached position.

    `cps` is the cache's list of top candidate evals (best first, side-to-move
    centipawns). `n_legal` is the legal move count. `complexity` is the primary
    metric (decision entropy); the rest are robustness variants read off the
    same row. A single-candidate or empty list is a forced, trivial position.
    """
    if not cps:
        return {
            "complexity": 0.0,
            "eval_gap_1_2": None,
            "n_reasonable": 1,
            "is_only_move": True,
        }
    feats = complexity_features(cps, n_legal)
    return {
        "complexity": feats["decision_entropy"],
        "eval_gap_1_2": feats["eval_gap_1_2"],
        "n_reasonable": feats["n_reasonable"],
        "is_only_move": feats["is_only_move"],
    }


def complexity_frame(cache: pd.DataFrame) -> pd.DataFrame:
    """Turn raw cache rows (fen, cps, n_legal) into per-fen complexity columns."""
    rows = [position_complexity(list(c), int(n)) for c, n in zip(cache["cps"], cache["n_legal"])]
    out = pd.DataFrame(rows)
    out.insert(0, "fen", cache["fen"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Ordinary least squares with basic stats, since statsmodels is not installed
# ---------------------------------------------------------------------------

def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """Closed-form OLS. X must already include an intercept column.

    Returns coefficients and, for each, the standard error, t value, and
    two-sided p value, plus the model R squared. Standard textbook formulas,
    good enough here since we only need effect size and significance, not a
    full modelling toolkit.
    """
    from scipy import stats

    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    # pinv, not inv, so a rank-deficient subset (a constant dummy after slicing)
    # degrades gracefully instead of throwing.
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"beta": beta, "se": se, "t": t, "p": p, "r2": r2, "dof": dof, "n": n}


def standardized_complexity_effect(df: pd.DataFrame) -> dict:
    """Partial effect of complexity on log think-time, holding the confounds.

    Builds a design matrix with a z-scored complexity plus z-scored move number
    and log clock, and one-hot dummies for phase, regime, and rating band. The
    complexity coefficient is then the change in log think-time per one standard
    deviation of complexity, with the listed confounds held fixed.
    """
    y = df[TARGET].to_numpy(dtype=float)

    def z(col):
        v = df[col].to_numpy(dtype=float)
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v * 0.0

    cols = [np.ones(len(df))]  # intercept
    names = ["intercept"]
    for c in NUMERIC_FEATURES:
        cols.append(z(c))
        names.append(c)
    for c in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
        for name in dummies.columns:
            cols.append(dummies[name].to_numpy(dtype=float))
            names.append(name)

    X = np.column_stack(cols)
    fit = ols_fit(X, y)
    ci = names.index("complexity")
    return {
        "names": names,
        "complexity_beta": float(fit["beta"][ci]),
        "complexity_se": float(fit["se"][ci]),
        "complexity_t": float(fit["t"][ci]),
        "complexity_p": float(fit["p"][ci]),
        "r2": fit["r2"],
        "n": fit["n"],
    }


# ---------------------------------------------------------------------------
# The saved expected-think-time mapping
# ---------------------------------------------------------------------------

def build_thinktime_model():
    """Pipeline that predicts log(1 + think-time seconds) from the features.

    One-hot the categoricals, pass the numerics through, then a plain linear
    fit. Kept as an sklearn Pipeline so later steps just load and call predict
    with the same column names, no hand-rolled encoding to keep in sync.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("lin", LinearRegression())])


def fit_expected_thinktime(df: pd.DataFrame):
    """Fit the mapping on a frame that has the feature columns and TARGET."""
    model = build_thinktime_model()
    model.fit(df[CATEGORICAL_FEATURES + NUMERIC_FEATURES], df[TARGET])
    return model


def expected_thinktime(model, df: pd.DataFrame) -> np.ndarray:
    """Expected think-time in seconds for each row (inverse of the log target)."""
    pred_log = model.predict(df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    return np.expm1(pred_log).clip(min=0.0)


def thinktime_residual(model, df: pd.DataFrame) -> np.ndarray:
    """Observed minus expected think-time in seconds. This is 'time pressure':
    positive means the player spent longer than the position and clock warrant,
    negative means they moved faster than expected."""
    return df["time_spent_s"].to_numpy(dtype=float) - expected_thinktime(model, df)


def save_mapping(model, curve: pd.DataFrame, spec: dict, out_dir: str | Path) -> dict:
    """Persist the fitted model, the complexity-to-time curve, and a spec file."""
    import joblib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "expected_thinktime_model.joblib"
    curve_path = out_dir / "complexity_thinktime_curve.parquet"
    spec_path = out_dir / "expected_thinktime_spec.json"

    joblib.dump(model, model_path)
    curve.to_parquet(curve_path, index=False)
    spec_path.write_text(json.dumps(spec, indent=2, default=str))
    return {"model": str(model_path), "curve": str(curve_path), "spec": str(spec_path)}


def load_mapping(out_dir: str | Path):
    """Load the fitted model back for later steps."""
    import joblib

    return joblib.load(Path(out_dir) / "expected_thinktime_model.joblib")


# ---------------------------------------------------------------------------
# Think-time from a clock series (thin wrapper so the sign is obvious)
# ---------------------------------------------------------------------------

def log1p_seconds(x) -> float:
    return math.log1p(max(0.0, float(x)))


# ---------------------------------------------------------------------------
# Data assembly (shared by the STEP 1 validation and the resolution pilot)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def load_complexity(cache_dir: str | Path) -> pd.DataFrame:
    """Read every cache batch under a dir and turn it into per-fen complexity.

    Works on either cache (the existing 30k/multipv-4 one or a pilot cache),
    since both store the same fen/cps/n_legal schema. The number of candidates
    in `cps` follows whatever multipv that cache was built at.
    """
    batches = sorted(glob.glob(str(Path(cache_dir) / "batch_*.parquet")))
    _log(f"reading {len(batches)} cache batches from {cache_dir} ...")
    cache = pd.concat((pd.read_parquet(b) for b in batches), ignore_index=True)
    cache = cache.drop_duplicates("fen")
    _log(f"  cached unique FENs: {len(cache):,}")
    return complexity_frame(cache)


def load_thinktime(interim_dir: str | Path, players: set[str]) -> pd.DataFrame:
    """All moves by the given players, with per-move think-time filled in.

    Think-time needs the full clock series for each side of a game, so we pull
    every ply the players made (book moves included), compute the series, and
    let the caller keep only the plies it wants afterwards.
    """
    shards = sorted(glob.glob(str(Path(interim_dir) / "*" / "shard_*.parquet")))
    _log(f"scanning {len(shards)} interim shards for {len(players):,} players ...")
    kept = []
    for sh in shards:
        df = pd.read_parquet(sh, columns=INTERIM_COLS)
        df = df[df["player_id"].isin(players)]
        if len(df):
            kept.append(df)
    moves = pd.concat(kept, ignore_index=True)
    _log(f"  sample moves (all, incl. book): {len(moves):,}")

    # Base and increment straight off the TimeControl header.
    tc = moves["time_control"].str.split("+", expand=True)
    moves["base"] = pd.to_numeric(tc[0], errors="coerce")
    moves["inc"] = pd.to_numeric(tc[1], errors="coerce")
    moves = moves.dropna(subset=["base", "inc"])

    # Think-time = prev same-side clock - this clock + increment, base as the
    # first previous clock, clamped at 0. Same rule as timespent.py, vectorized.
    moves = moves.sort_values(["game_id", "player_id", "ply"])
    prev = moves.groupby(["game_id", "player_id"], sort=False)["clock_s"].shift(1)
    prev = prev.fillna(moves["base"])
    moves["clock_before_s"] = prev
    moves["time_spent_s"] = (prev - moves["clock_s"] + moves["inc"]).clip(lower=0)

    # Regime from the actual TimeControl, not the event name.
    regime_map = {tcv: classify_time_control(tcv) for tcv in moves["time_control"].unique()}
    moves["regime"] = moves["time_control"].map(regime_map)
    return moves[["game_id", "player_id", "ply", "move_number", "time_spent_s",
                  "clock_before_s", "regime"]]


def assemble(cfg: dict, proc: str | Path) -> pd.DataFrame:
    """Join complexity, think-time, phase, and rating band onto the clean set.

    One row per clean sample move, carrying fen_before, observed think-time, the
    confounds (phase, move number, clock remaining, regime, rating band), and the
    existing-cache complexity. This is the frame the STEP 1 validation runs on,
    and the pilot filters it to its sampled FENs.
    """
    proc = Path(proc)
    tierb = proc / "tierb"
    sample = pd.read_parquet(tierb / "sample_players.parquet")[["player_id", "band", "glicko"]]
    players = set(sample["player_id"])
    _log(f"sample players: {len(players):,}")

    clean = pd.read_parquet(tierb / "moves_with_fen.parquet")
    _log(f"clean sample positions: {len(clean):,}")

    comp = load_complexity(tierb / "cache")
    think = load_thinktime(Path(cfg["paths"]["interim"]) / "moves", players)

    df = clean.merge(think, on=["game_id", "player_id", "ply"], how="left")
    n_before = len(df)
    df = df.dropna(subset=["time_spent_s"])
    _log(f"joined think-time: {len(df):,} / {n_before:,} clean positions matched")

    df = df.merge(comp, left_on="fen_before", right_on="fen", how="left")
    matched = df["complexity"].notna().sum()
    _log(f"joined complexity: {matched:,} / {len(df):,} have a cached complexity")
    df = df.dropna(subset=["complexity"])

    df = df.merge(sample, on="player_id", how="left")

    # Phase from the FEN, same rule as features.game_phase, vectorized.
    board = df["fen_before"].str.split(" ", n=1).str[0]
    minors_majors = board.str.count("[nbrqNBRQ]")
    df["game_phase"] = np.where(
        df["ply"] <= cfg["book_plies"], "opening",
        np.where(minors_majors <= 6, "endgame", "middlegame"),
    )

    # Drop any premove that slipped through, then build the model columns.
    df = df[df["time_spent_s"] >= MIN_THINK_S]
    df["log_time_spent"] = np.log1p(df["time_spent_s"].astype(float))
    df["log_clock_before"] = np.log1p(df["clock_before_s"].clip(lower=0).astype(float))
    df["rating_band"] = df["band"]
    _log(f"final analysis rows: {len(df):,}")
    return df


def spearman(x, y) -> tuple[float, float, int]:
    """Spearman rho, p, and n over the finite pairs. NaN if too few points."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan"), float("nan"), int(m.sum())
    rho, p = stats.spearmanr(x[m], y[m])
    return float(rho), float(p), int(m.sum())
