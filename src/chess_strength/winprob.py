"""Phase 4: move quality on the win-probability scale.

Centipawns are not linear in how much a move matters. A swing from +900 to
+800 barely changes who is winning, but +50 to -50 flips the game. Lichess maps
an eval to a win percentage with a sigmoid, and judges a move by how much win%
it threw away. We copy their formulas exactly so our numbers line up with what
players already see on the site.

These are pure functions on centipawn / win% values. Lining up the eval before
and after each move (they arrive in different points of view) is a join that
happens in Phase 6, not here.
"""

from __future__ import annotations

import math

# Lichess win% sigmoid constant. Mirrors config/default.yaml winprob_k. Kept as
# a default so callers can use these functions without threading config through.
WINPROB_K = 0.00368208
EVAL_CLIP_CP = 1000


def win_pct(cp: float, *, clip_cp: int = EVAL_CLIP_CP, k: float = WINPROB_K) -> float:
    """Win percentage in [0, 100] for a centipawn eval from the mover's POV.

    A dead-equal position is 50. The eval is clipped first so a crushing but
    already-won position does not dominate.
    """
    c = max(-clip_cp, min(clip_cp, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-k * c)) - 1.0)


def wpl(win_before: float, win_after: float) -> float:
    """Win-percentage lost by a move: win% before minus win% after.

    Both inputs are the mover's win%. A move that improves the position gets no
    negative credit, so we clamp at 0. This, not raw centipawn loss, is our
    primary move-quality target.
    """
    return max(0.0, win_before - win_after)


def accuracy(win_before: float, win_after: float) -> float:
    """Lichess move Accuracy% for a win% drop, clamped to [0, 100].

    A reference metric only. Our modelling uses WPL; accuracy is here so results
    can be read against what the site reports.
    """
    delta = max(0.0, win_before - win_after)
    acc = 103.1668 * math.exp(-0.04354 * delta) - 3.1669
    return max(0.0, min(100.0, acc))
