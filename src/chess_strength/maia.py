"""Maia human-move dispersion as a position-complexity metric.

The engine-eval entropy from STEP 1 saturates and turns over: "several engine
moves within a pawn" mixes genuinely hard positions with dull equal ones, and
more engine depth did not fix it (the resolution pilot was NO-GO). Maia measures
something different: how HUMANS at a rating band spread their move choice. If
humans strongly agree on one move the position is easy, if they split across many
moves it is hard. That is a human-difficulty signal, not an engine-agreement one.

Dispersion here is read off Maia's move probability distribution for a position
at a given rating band. We keep three readings so the analysis can compare them:

- maia_entropy: raw Shannon entropy of the distribution (nats). The primary
  reading. Higher = human choice is more spread = harder.
- maia_entropy_norm: entropy divided by log(number of moves with any weight), in
  [0, 1]. Controls for how many moves carry probability, the same normalized
  shape STEP 1 used for the engine metric, so the two are comparable.
- maia_top1: probability mass on the single most likely move. Low = dispersed.

This module is pure math on a probability dict so it imports cleanly in either
the main pipeline env or the separate Maia inference env.
"""

from __future__ import annotations

import math

# Rating band label to a representative Elo for Maia conditioning. Midpoints,
# with the open top band capped where Maia's range flattens out.
BAND_ELO = {
    "[0, 1300)": 1200,
    "[1300, 1500)": 1400,
    "[1500, 1700)": 1600,
    "[1700, 1900)": 1800,
    "[1900, 2100)": 2000,
    "[2100, 9999)": 2200,
}


def band_to_elo(band: str) -> int:
    """Representative Elo for a rating band label, with a midpoint fallback."""
    if band in BAND_ELO:
        return BAND_ELO[band]
    # Fallback: parse "[lo, hi)" and take a midpoint, so an unseen label still works.
    try:
        lo, hi = band.strip("[)").split(",")
        lo, hi = int(lo), int(hi)
        hi = min(hi, 2400)
        return int((lo + hi) / 2)
    except (ValueError, AttributeError, IndexError):
        return 1600


def move_dispersion(move_probs: dict) -> dict:
    """Dispersion of a Maia move distribution. High = humans disagree = hard.

    `move_probs` maps a move (uci) to its Maia probability. Maia's outputs may
    not sum to exactly 1, so we renormalize over the moves that carry weight.
    A single-move or empty distribution is trivially easy (dispersion 0).
    """
    ps = [float(p) for p in move_probs.values() if p and p > 0]
    tot = sum(ps)
    n = len(ps)
    if tot <= 0 or n == 0:
        return {"maia_entropy": 0.0, "maia_entropy_norm": 0.0, "maia_top1": 1.0, "maia_n_moves": n}
    ps = [p / tot for p in ps]
    ent = -sum(p * math.log(p) for p in ps)
    return {
        "maia_entropy": ent,
        "maia_entropy_norm": ent / math.log(n) if n > 1 else 0.0,
        "maia_top1": max(ps),
        "maia_n_moves": n,
    }
