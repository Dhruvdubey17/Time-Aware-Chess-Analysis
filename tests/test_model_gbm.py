"""Phase 8 tests: the GBM plumbing on small synthetic data.

Checks that build_matrix keeps the target out of the features, that a model
trained on a clear signal recovers it (high rank correlation, finite MAE), and
that importance ranks the real driver first. Real-data metrics come from the
Phase 8 runner on the two-month table.
"""

import numpy as np
import pandas as pd

from chess_strength.model_gbm import (
    build_matrix,
    evaluate_model,
    feature_importance,
    train_gbm,
)


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)
    # Glicko driven mostly by x0, a little by x1.
    glicko = 1500 + 300 * x0 + 50 * x1 + 100 * noise
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "mean_wpl": x0, "mean_time": x1, "n_clean": rng.integers(50, 500, n),
        "glicko": glicko,
    })


def test_build_matrix_excludes_target_and_keys():
    df = _synthetic(50)
    X, y, cols = build_matrix(df)
    assert "glicko" not in cols and "player_id" not in cols and "n_clean" not in cols
    assert set(cols) == {"mean_wpl", "mean_time"}
    assert np.allclose(y, df["glicko"].to_numpy())
    assert X.shape == (50, 2)


def test_gbm_recovers_signal():
    df = _synthetic(1800)
    ids = df["player_id"].to_numpy()
    rng = np.random.default_rng(1)
    r = rng.random(len(ids))
    tr, va, te = df[r < 0.6], df[(r >= 0.6) & (r < 0.8)], df[r >= 0.8]

    Xtr, ytr, cols = build_matrix(tr)
    Xva, yva, _ = build_matrix(va, cols)
    Xte, yte, _ = build_matrix(te, cols)

    model = train_gbm(Xtr, ytr, Xva, yva, params={"n_estimators": 200})
    res = evaluate_model(model, Xte, yte)
    # A strong linear signal should give a high rank correlation.
    assert res["spearman"] > 0.8
    assert res["mae"] < 150

    imp = feature_importance(model, cols)
    assert imp.iloc[0]["feature"] == "mean_wpl"
