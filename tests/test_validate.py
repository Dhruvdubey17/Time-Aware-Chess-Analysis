"""Offline checks for the STEP 5 validation statistics (synthetic data)."""

import numpy as np

from chess_strength import validate as v


def test_spearman_ci_positive():
    x = np.arange(200.0)
    y = x + np.random.default_rng(0).normal(0, 5, 200)
    r = v.spearman_ci(x, y)
    assert r["rho"] > 0.9 and r["lo"] > 0 and r["n"] == 200


def test_rating_monotonicity_detects_signal():
    rng = np.random.default_rng(0)
    rating = rng.uniform(1000, 2200, 500)
    # Upgrade rate rises with rating plus noise.
    rate = (rating - 1000) / 1200 * 0.1 + rng.normal(0, 0.01, 500)
    out = v.rating_monotonicity(rate, rating)
    assert out["rho"] > 0.5 and out["lo"] > 0


def test_within_position_concordance():
    rng = np.random.default_rng(0)
    # 300 positions, 4 players each. Stronger players get upgraded.
    pos, rating, up = [], [], []
    for p in range(300):
        ratings = rng.uniform(1000, 2200, 4)
        thresh = np.median(ratings)
        for r in ratings:
            pos.append(p); rating.append(r); up.append(r > thresh)
    out = v.within_position_skill(pos, rating, up, n_boot=300, seed=0)
    assert out["c"] > 0.8 and out["lo"] > 0.5 and out["n_positions"] > 100

    # Null: upgrade unrelated to rating -> concordance near 0.5.
    up_rand = rng.random(len(pos)) > 0.5
    null = v.within_position_skill(pos, rating, up_rand, n_boot=300, seed=0)
    assert 0.4 < null["c"] < 0.6


def test_partial_rank_correlation():
    rng = np.random.default_rng(0)
    n = 2000
    difficulty = rng.normal(0, 1, n)
    rating = 0.5 * difficulty + rng.normal(0, 1, n)     # rating correlates with difficulty
    # Upgrade driven by difficulty plus a real rating effect.
    up = (0.6 * difficulty + 0.4 * rating + rng.normal(0, 1, n)) > 0.5
    out = v.partial_rank_correlation(up, rating, [difficulty])
    assert out["rho"] > 0.1 and out["p"] < 0.01
    # No rating effect beyond difficulty -> partial near zero.
    up2 = (0.8 * difficulty + rng.normal(0, 1, n)) > 0.0
    out2 = v.partial_rank_correlation(up2, rating, [difficulty])
    assert abs(out2["rho"]) < 0.1


def test_predictive_validity_partial():
    rng = np.random.default_rng(0)
    n = 400
    quality = rng.normal(0, 1, n)
    upgrade = 0.5 * quality + rng.normal(0, 1, n)   # correlated with quality
    # Future depends on upgrade beyond quality.
    future = 0.3 * quality + 0.4 * upgrade + rng.normal(0, 1, n)
    out = v.predictive_validity(future, upgrade, quality)
    assert out["partial_rho"] > 0.15 and out["p"] < 0.01

    # Upgrade adds nothing beyond quality -> partial near 0.
    upgrade_noise = rng.normal(0, 1, n)
    future2 = 0.5 * quality + rng.normal(0, 1, n)
    out2 = v.predictive_validity(future2, upgrade_noise, quality)
    assert abs(out2["partial_rho"]) < 0.15
