"""Phase 8: LightGBM player-strength model (framing i).

Regress per-player aggregate features onto the player's Glicko. The features are
summaries of move quality, clock usage, game phase, and position sharpness over
the whole two-month history. Clock and complexity are allowed here as predictive
inputs; this is a learner, not the causal residual of Phase 7 (standing rule 14).

The target Glicko is never a feature. build_matrix drops the id and target
columns and returns everything else as X.
"""

from __future__ import annotations

import pandas as pd

# Columns that are keys or the target, never features.
NON_FEATURES = {"player_id", "glicko", "n_clean", "n_games"}

# Feature groups for the Phase 9 ablation. CLOCK is how a player uses the clock
# (time spent, time left, time pressure), including per-phase time. Time-control
# settings (base time, increment) and regime shares stay in the backbone, so a
# no-clock arm still knows the time control, just not the usage. That isolates
# the clock-usage signal, which is the thesis. TIER_A_COMPLEXITY is the two
# population-wide eval-shape proxies; engine MultiPV complexity is not a
# per-player feature here (the cache covers too few players).
CLOCK_FEATURES = [
    "mean_log_time_spent", "mean_time_spent", "mean_clock_remaining",
    "time_pressure_rate", "mean_frac_time_used", "time_middlegame", "time_endgame",
]
TIER_A_COMPLEXITY_FEATURES = ["mean_eval_swing", "mean_eval_volatility"]

# Tier-B engine (MultiPV) complexity, per-player aggregates of the four
# per-position features derived from Stockfish top-k evals. Only available on the
# annotated sample, so it is used only in the Phase 9 Tier-B ablation.
TIERB_COMPLEXITY_FEATURES = [
    "tb_gap_mean", "tb_gap_std", "tb_nreason_mean", "tb_nreason_std",
    "tb_entropy_mean", "tb_entropy_std", "tb_onlymove_rate",
]


def features_excluding(all_cols: list[str], drop: list[str]) -> list[str]:
    """Feature list with a group removed, order preserved (base minus a group)."""
    drop = set(drop)
    return [c for c in all_cols if c not in drop]


def build_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None):
    """Split a per-player table into (X, y, feature_cols).

    Everything that is not an id, the target, or a raw count is a feature unless
    an explicit list is given. Counts are dropped by default so strength is not
    read off activity.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    return df[feature_cols].to_numpy(float), df["glicko"].to_numpy(float), feature_cols


def train_gbm(X_tr, y_tr, X_val, y_val, params: dict | None = None):
    """Train a LightGBM regressor with early stopping on a player-disjoint val set."""
    import lightgbm as lgb

    p = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "random_state": 0,
        "n_jobs": -1,
    }
    if params:
        p.update(params)

    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def evaluate_model(model, X, y) -> dict:
    """MAE (Elo) and Spearman rho of predictions against Glicko."""
    from .evaluate import mae, spearman

    pred = model.predict(X)
    return {"mae": mae(pred, y), "spearman": spearman(pred, y), "n": len(y)}


def feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """Gain-based importance per feature, most important first."""
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    out = pd.DataFrame({"feature": feature_cols, "gain": gain})
    out["gain_pct"] = 100 * out["gain"] / out["gain"].sum()
    return out.sort_values("gain", ascending=False).reset_index(drop=True)
