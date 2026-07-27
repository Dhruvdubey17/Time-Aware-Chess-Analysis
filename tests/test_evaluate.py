"""Tests for the shared split and metrics."""

import numpy as np

from chess_strength.evaluate import mae, spearman, split_players


def test_split_is_player_disjoint_and_covers_all():
    ids = [f"p{i}" for i in range(1000)]
    assign = split_players(ids, seed=0)
    groups = {"train": set(), "val": set(), "test": set()}
    for pid, g in assign.items():
        groups[g].add(pid)
    # No player in two groups, every player placed.
    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["val"] & groups["test"] == set()
    assert sum(len(s) for s in groups.values()) == 1000


def test_split_fractions_roughly_hold_and_deterministic():
    ids = [f"p{i}" for i in range(5000)]
    a1 = split_players(ids, seed=0, test_frac=0.3, val_frac=0.15)
    a2 = split_players(ids, seed=0, test_frac=0.3, val_frac=0.15)
    assert a1 == a2  # deterministic
    frac_test = sum(v == "test" for v in a1.values()) / 5000
    frac_val = sum(v == "val" for v in a1.values()) / 5000
    assert abs(frac_test - 0.30) < 0.03
    assert abs(frac_val - 0.15) < 0.03


def test_test_set_matches_phase7_draw():
    # Phase 7 defined test as the first random draw < 0.3 on sorted ids with the
    # same seed. Phase 8 must reuse that exact held-out set.
    ids = [f"p{i}" for i in range(2000)]
    sorted_ids = np.sort(np.asarray(ids))
    r = np.random.default_rng(0).random(len(sorted_ids))
    expected_test = set(sorted_ids[r < 0.3])
    got_test = {p for p, g in split_players(ids, seed=0).items() if g == "test"}
    assert got_test == expected_test


def test_metrics():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert mae([1, 2], [2, 4]) == 1.5
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
