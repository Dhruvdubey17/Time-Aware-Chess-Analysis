"""Shared evaluation helpers: the player split and the metrics.

The split is by player, never by move, so the same player never lands on both
sides. A player who appears in more than one month is one player (identity is the
username), so this split handles the month-disjoint requirement on its own once
the per-player table is built.

The test assignment matches Phase 7: with the same seed the first random draw
reproduces the same held-out set, so Phase 8 is judged on the same players as the
Phase 7 gate. Phase 8 also needs a validation slice for early stopping, carved
out of the non-test players.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_players(
    player_ids, seed: int = 0, test_frac: float = 0.3, val_frac: float = 0.15
) -> dict:
    """Assign each player to train, val, or test. Player-disjoint by construction.

    test = the same held-out third as Phase 7 (same seed, same first draw). val is
    the next slice, the rest is train.
    """
    ids = np.sort(np.asarray(player_ids))
    r = np.random.default_rng(seed).random(len(ids))
    assign = np.where(
        r < test_frac, "test", np.where(r < test_frac + val_frac, "val", "train")
    )
    return dict(zip(ids, assign))


def mae(pred, true) -> float:
    """Mean absolute error, in the same units as the target (Elo here)."""
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(true, float))))


def spearman(pred, true) -> float:
    """Spearman rank correlation between predictions and truth."""
    return float(pd.Series(np.asarray(pred, float)).corr(
        pd.Series(np.asarray(true, float)), method="spearman"))
