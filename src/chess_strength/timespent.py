"""Phase 4: how long each move took.

The `[%clk]` tag is the time left on the mover's own clock after the move, with
the increment already added back. So the time spent on a move is how far the
clock dropped since that same player's previous move, plus the increment they
just earned:

    time_spent = prev_same_side_clock - this_clock + increment

The first move of each side has no previous clock, so we use the base time. In
increment games the clock can rise across moves (you gain more than you spend);
the formula handles that on its own, and we clamp tiny negative values from
rounding to zero.

Reference: kraktus/lichess-time-spent.
"""

from __future__ import annotations


def parse_time_control(tc: str) -> tuple[int, int]:
    """Split a Lichess TimeControl like "180+2" into (base_seconds, increment).

    Raises ValueError on anything that is not base+increment, e.g. the "-"
    used for correspondence, so a bad header fails loudly instead of silently
    producing wrong times.
    """
    base, sep, inc = tc.partition("+")
    if not sep or not base.strip().isdigit() or not inc.strip().isdigit():
        raise ValueError(f"unexpected TimeControl: {tc!r}")
    return int(base), int(inc)


def time_spent_series(
    clocks: list[float], base_time: float, increment: float
) -> list[float]:
    """Per-move time spent for one side's clocks, in move order.

    `clocks` is that side's `[%clk]` readings for the game. The previous clock
    for the first move is the base time.
    """
    out: list[float] = []
    prev = base_time
    for clk in clocks:
        out.append(max(0.0, prev - clk + increment))
        prev = clk
    return out


def add_time_spent(rows: list[dict]) -> list[dict]:
    """Add a `time_spent_s` field to per-move rows from parse_moves.

    Rows are grouped by game and color and must already be in ply order (as
    parse_moves emits them). Each group runs through time_spent_series with the
    base and increment read from that game's TimeControl.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["game_id"], row["color"]), []).append(row)

    for group in groups.values():
        base, inc = parse_time_control(group[0]["time_control"])
        spent = time_spent_series([r["clock_s"] for r in group], base, inc)
        for row, t in zip(group, spent):
            row["time_spent_s"] = t

    return rows
