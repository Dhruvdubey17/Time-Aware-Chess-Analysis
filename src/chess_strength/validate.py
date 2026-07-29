"""STEP 5: statistics for validating that upgrades track genuine skill.

Rating is the thing we validate against. It never enters the classifier. These
helpers take the finished per-move labels plus held-out rating and ask three
questions:

- rating_monotonicity: do stronger players get upgraded more often?
- within_position_skill: on the SAME position, do stronger players get more
  upgrades? This controls for position difficulty by construction, so it is the
  cleanest test. We measure a within-position concordance: over all pairs of
  players who faced the same position where one was upgraded and one was not, how
  often was the upgraded one the higher rated. 0.5 means no link, above 0.5 means
  stronger players are upgraded more on identical positions.
- predictive_validity: does past upgrade frequency predict a player's future
  rating beyond their past move quality (partial correlation)?

Pure numpy and scipy so it is unit testable on synthetic data.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def spearman_ci(x, y, alpha: float = 0.05) -> dict:
    """Spearman rho with a Fisher z confidence interval and n."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 5:
        return {"rho": float("nan"), "p": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": n}
    rho, p = stats.spearmanr(x[m], y[m])
    # Fisher z interval, the standard large-sample approximation for rho.
    z = math.atanh(max(-0.999999, min(0.999999, rho)))
    se = 1.0 / math.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    lo, hi = math.tanh(z - zc * se), math.tanh(z + zc * se)
    return {"rho": float(rho), "p": float(p), "lo": float(lo), "hi": float(hi), "n": n}


def rating_monotonicity(rate: np.ndarray, rating: np.ndarray) -> dict:
    """TEST 1: correlation between per-player upgrade rate and rating."""
    return spearman_ci(rating, rate)


def within_position_skill(pos_key, rating, upgraded, n_boot: int = 1000, seed: int = 0) -> dict:
    """TEST 2: within-position concordance between upgrade and rating.

    For each position faced by several players, count pairs where exactly one of
    the two was upgraded, and score the pair concordant if the upgraded player was
    the higher rated. The c-statistic is concordant / (concordant + discordant),
    with a bootstrap CI resampling positions (the unit of independence).
    """
    import pandas as pd

    df = pd.DataFrame({"pos": np.asarray(pos_key), "r": np.asarray(rating, dtype=float),
                       "up": np.asarray(upgraded).astype(bool)})

    def group_pairs(g):
        up = g.loc[g["up"], "r"].to_numpy()
        dn = g.loc[~g["up"], "r"].to_numpy()
        if len(up) == 0 or len(dn) == 0:
            return 0.0, 0.0
        # Concordant: upgraded rating > non-upgraded rating. Ties count as half.
        diff = up[:, None] - dn[None, :]
        conc = float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
        return conc, float(diff.size)

    per_pos = []
    for _, g in df.groupby("pos"):
        if g["up"].nunique() < 2:  # needs at least one up and one not
            continue
        conc, tot = group_pairs(g)
        if tot > 0:
            per_pos.append((conc, tot))
    if not per_pos:
        return {"c": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_positions": 0, "n_pairs": 0}
    per_pos = np.array(per_pos)
    conc_tot, pair_tot = per_pos[:, 0].sum(), per_pos[:, 1].sum()
    c = conc_tot / pair_tot

    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(len(per_pos))
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        s = per_pos[take]
        boots.append(s[:, 0].sum() / s[:, 1].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"c": float(c), "lo": float(lo), "hi": float(hi),
            "n_positions": len(per_pos), "n_pairs": int(pair_tot)}


def partial_rank_correlation(y, x, controls: list) -> dict:
    """Rank-based partial correlation of x and y after removing the controls.

    Used as the feasible substitute for TEST 2 when exact positions do not recur:
    does rating (x) predict upgrade (y) at the move level once difficulty and
    clock context are held. Move-level n is large, so read the effect size, not
    just the tiny p.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ctrls = [np.asarray(c, dtype=float) for c in controls]
    m = np.isfinite(y) & np.isfinite(x)
    for c in ctrls:
        m &= np.isfinite(c)
    n = int(m.sum())
    if n < 20:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    Z = np.column_stack([np.ones(n)] + [stats.rankdata(c[m]) for c in ctrls])

    def resid(v):
        beta, _, _, _ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    r, p = stats.pearsonr(resid(stats.rankdata(x[m])), resid(stats.rankdata(y[m])))
    return {"rho": float(r), "p": float(p), "n": n}


def _resid(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residual of y after a simple linear fit on z (with intercept)."""
    X = np.column_stack([np.ones_like(z), z])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def predictive_validity(future: np.ndarray, past_upgrade: np.ndarray,
                        past_quality: np.ndarray) -> dict:
    """TEST 3: partial correlation of future rating with past upgrade rate,
    controlling for past move quality. Rank-based so it matches the Spearman
    framing: correlate the residuals after removing past_quality from both."""
    future = np.asarray(future, dtype=float)
    up = np.asarray(past_upgrade, dtype=float)
    q = np.asarray(past_quality, dtype=float)
    m = np.isfinite(future) & np.isfinite(up) & np.isfinite(q)
    n = int(m.sum())
    if n < 10:
        return {"partial_rho": float("nan"), "p": float("nan"), "n": n,
                "raw_rho": float("nan")}
    fr = stats.rankdata(future[m])
    ur = stats.rankdata(up[m])
    qr = stats.rankdata(q[m])
    rf = _resid(fr, qr)
    ru = _resid(ur, qr)
    r, p = stats.pearsonr(ru, rf)
    raw = stats.spearmanr(up[m], future[m]).correlation
    return {"partial_rho": float(r), "p": float(p), "n": n, "raw_rho": float(raw)}
