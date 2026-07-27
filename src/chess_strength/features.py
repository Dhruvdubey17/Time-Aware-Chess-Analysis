"""Phase 6: assemble one clean per-move feature table and a per-player table.

This is where the pieces from earlier phases meet. Each parsed move gets its
time-spent (Phase 4), its win-probability quality (Phase 4), and its Tier-A
complexity (Phase 5) joined on, plus the flags we filter by later: in_book for
opening theory and premove_suspect for moves played too fast to be real
decisions.

Two things need a per-game pass in ply order and cannot be done row by row: the
win% before a move (it is the position the mover faced, which lives on the
previous ply and in a different point of view) and the opponent's clock (their
most recent reading). Everything else is local to a row.

The player's own rating is kept in the move table only as a conditioning signal
and is labelled as such. The real target, their Glicko, is held aside in the
player table and is never a feature.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .complexity import add_tier_a
from .stream_filter import classify_time_control
from .timespent import add_time_spent, parse_time_control
from .winprob import win_pct, wpl

# Rough piece values in centipawns, for a material-balance feature.
_PIECE_CP = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900}

# Below this many seconds on the clock we call it time pressure.
TIME_PRESSURE_S = 30.0


def _piece_counts(fen: str) -> str:
    """Just the board part of a FEN, for counting pieces."""
    return fen.split(" ", 1)[0]


def material_cp(fen: str, mover_is_white: bool) -> int:
    """Material balance from the mover's point of view, in centipawns."""
    board = _piece_counts(fen)
    white = sum(_PIECE_CP[c.lower()] * board.count(c) for c in "PNBRQ")
    black = sum(_PIECE_CP[c] * board.count(c) for c in "pnbrq")
    diff = white - black
    return diff if mover_is_white else -diff


def game_phase(fen: str, ply: int, book_plies: int) -> str:
    """Opening while still in book, endgame once few pieces remain, else middlegame."""
    if ply <= book_plies:
        return "opening"
    board = _piece_counts(fen)
    minors_majors = sum(board.count(c) for c in "nbrqNBRQ")
    return "endgame" if minors_majors <= 6 else "middlegame"


def eval_bucket(win_before: float) -> str:
    """Coarse band for where the mover stood before the move, on the win% scale."""
    if win_before < 20:
        return "losing_clear"
    if win_before < 40:
        return "worse"
    if win_before <= 60:
        return "equal"
    if win_before <= 80:
        return "better"
    return "winning_clear"


def assemble_features(rows: list[dict], cfg: dict) -> list[dict]:
    """Join every per-move signal into one flat feature row per move.

    `rows` are parse_moves records for one or more games. They are enriched in
    place with time-spent and Tier-A complexity first, then walked per game in
    ply order for the two signals that need history.
    """
    clip = cfg["eval_clip_cp"]
    k = cfg["winprob_k"]
    book_plies = cfg["book_plies"]
    premove_thresh = cfg["premove_time_threshold_s"]

    add_time_spent(rows)
    add_tier_a(rows)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["game_id"], []).append(row)

    out: list[dict] = []
    for group in groups.values():
        group = sorted(group, key=lambda r: r["ply"])
        base, inc = parse_time_control(group[0]["time_control"])
        regime = classify_time_control(group[0]["time_control"])

        # Startpos is roughly equal, so the eval before the first move is 0.
        prev_white_cp = 0
        last_clock = {"white": None, "black": None}

        for row in group:
            is_white = row["color"] == "white"

            before_cp = prev_white_cp if is_white else -prev_white_cp
            after_cp = row["eval_cp"]
            win_before = win_pct(before_cp, clip_cp=clip, k=k)
            win_after = win_pct(after_cp, clip_cp=clip, k=k)

            opp_color = "black" if is_white else "white"
            opp_clock = last_clock[opp_color]
            if opp_clock is None:
                opp_clock = base

            clock_rem = row["clock_s"]
            frac_used = (base - clock_rem) / base if base > 0 else 0.0
            frac_used = min(1.0, max(0.0, frac_used))
            time_spent = row["time_spent_s"]

            out.append(
                {
                    # keys
                    "game_id": row["game_id"],
                    "player_id": row["player_id"],
                    "color": row["color"],
                    "ply": row["ply"],
                    "move_number": row["move_number"],
                    # clock
                    "time_spent_s": time_spent,
                    "log_time_spent": math.log1p(time_spent),
                    "clock_remaining_s": clock_rem,
                    "log_clock_remaining": math.log1p(max(0.0, clock_rem)),
                    "opp_clock_remaining_s": opp_clock,
                    "base_time": base,
                    "increment": inc,
                    "frac_time_used": frac_used,
                    "in_time_pressure": clock_rem < TIME_PRESSURE_S,
                    "regime": regime,
                    "is_blitz": regime == "blitz",
                    "is_rapid": regime == "rapid",
                    # quality
                    "win_pct_before": win_before,
                    "wpl": wpl(win_before, win_after),
                    "cpl_clipped": min(clip, max(0, before_cp - after_cp)),
                    "eval_bucket": eval_bucket(win_before),
                    # complexity (Tier A; Tier B is joined per FEN only where sampled)
                    "eval_swing": row["eval_swing"],
                    "eval_volatility": row["eval_volatility"],
                    "decisiveness": row["decisiveness"],
                    # structure
                    "game_phase": game_phase(row["fen_before"], row["ply"], book_plies),
                    "material": material_cp(row["fen_before"], is_white),
                    "is_check": row["san"].endswith(("+", "#")),
                    "is_capture": "x" in row["san"],
                    # context; player_rating is conditioning only, not the target
                    "player_rating": row["white_elo"] if is_white else row["black_elo"],
                    "opponent_rating": row["black_elo"] if is_white else row["white_elo"],
                    "in_book": row["ply"] <= book_plies,
                    "premove_suspect": time_spent < premove_thresh,
                }
            )

            prev_white_cp = row["eval_cp_white"]
            last_clock[row["color"]] = clock_rem

    return out


def build_players(features: pd.DataFrame) -> pd.DataFrame:
    """Per-player aggregate. Glicko is held aside as the target, never a feature."""
    g = features.groupby("player_id")
    players = g.agg(
        n_moves=("wpl", "size"),
        n_games=("game_id", "nunique"),
        mean_wpl=("wpl", "mean"),
        median_wpl=("wpl", "median"),
        wpl_dispersion=("wpl", "std"),
        glicko=("player_rating", "mean"),
    )
    # Single-move players have no dispersion; call it zero rather than NaN.
    players["wpl_dispersion"] = players["wpl_dispersion"].fillna(0.0)

    time_by_phase = features.pivot_table(
        index="player_id", columns="game_phase", values="time_spent_s", aggfunc="mean"
    ).rename(columns=lambda c: f"mean_time_{c}")

    return players.join(time_by_phase).reset_index()


def filter_clean(features: pd.DataFrame) -> pd.DataFrame:
    """View with book and premove-suspect moves dropped, for confounder checks."""
    return features[~features["in_book"] & ~features["premove_suspect"]]


def build_and_write(rows: list[dict], cfg: dict, out_dir: str | Path) -> dict[str, int]:
    """Assemble features, write moves_features.parquet and players.parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    moves = pd.DataFrame(assemble_features(rows, cfg))
    players = build_players(moves)

    moves.to_parquet(out_dir / "moves_features.parquet", index=False)
    players.to_parquet(out_dir / "players.parquet", index=False)
    return {"moves": len(moves), "players": len(players)}
