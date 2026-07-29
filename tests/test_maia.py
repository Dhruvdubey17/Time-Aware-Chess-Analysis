"""Offline checks for the Maia dispersion math (no model, no network)."""

import math

from chess_strength import maia


def test_move_dispersion_extremes():
    # Everyone agrees on one move -> low dispersion, top1 near 1.
    agree = maia.move_dispersion({"e2e4": 0.95, "d2d4": 0.05})
    # Humans split evenly across four moves -> high dispersion, top1 low.
    split = maia.move_dispersion({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})

    assert agree["maia_entropy"] < split["maia_entropy"]
    assert agree["maia_top1"] > split["maia_top1"]
    assert split["maia_top1"] == 0.25
    # Even split over n moves is maximum entropy, so normalized ~1.
    assert abs(split["maia_entropy_norm"] - 1.0) < 1e-9
    assert abs(split["maia_entropy"] - math.log(4)) < 1e-9


def test_move_dispersion_single_and_empty():
    assert maia.move_dispersion({"e2e4": 1.0})["maia_entropy"] == 0.0
    assert maia.move_dispersion({})["maia_top1"] == 1.0


def test_move_dispersion_renormalizes():
    # Maia outputs need not sum to 1; renormalizing must not change the shape.
    a = maia.move_dispersion({"x": 0.5, "y": 0.5})
    b = maia.move_dispersion({"x": 0.25, "y": 0.25})  # same shape, sums to 0.5
    assert abs(a["maia_entropy"] - b["maia_entropy"]) < 1e-9


def test_band_to_elo():
    assert maia.band_to_elo("[1500, 1700)") == 1600
    assert maia.band_to_elo("[2100, 9999)") == 2200
    # Unknown label falls back to a parsed midpoint (capped).
    assert maia.band_to_elo("[900, 1100)") == 1000
