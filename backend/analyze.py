"""Full review of one game: baseline labels and, when clocks allow it, the
time-aware re-classification.

This is the service port of scripts/classify_sample.py. It does not re-derive any
of the science; it reuses the validated pieces in chess_strength and wires them
together for a single game:

1. Read every position with local Stockfish (eval, entropy, only-move gap).
2. Fill in move quality (Win% loss) from `[%eval]` when present, else from the
   engine read.
3. When the game has clocks in a blitz or rapid time control, score human
   difficulty with local Maia and measure time pressure as the residual from the
   saved think-time mapping, then apply the calibrated upgrade rules.

Nothing here retunes a threshold. When a signal is missing the move keeps its
baseline label and the report says plainly why, never a faked upgrade.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import chess
import pandas as pd

from chess_strength import maia
from chess_strength import thinktime as tt
from chess_strength.classify import (
    ClassifyConfig,
    baseline_label,
    complexity_gate,
    is_brilliant,
    is_great,
    option_a_label,
    option_b_label,
    pressure_modifier,
    pressure_modifier_bullet,
    skill_of_move,
    win_pct_mate,
)
from chess_strength.features import material_cp

from . import book, engine, maia_client, premove
from .cache import FenCache
from .intake import GameReport

MATE_CP = 10000
DEFAULT_ELO = 1500  # rating-free fallback when a header has no Elo
# Glicko band cut points, the same edges the mapping and Maia bands were built on.
BANDS = [0, 1300, 1500, 1700, 1900, 2100, 9999]


def _service_workers(cfg: dict) -> int:
    return int(cfg.get("service_workers", 6))


def _cache_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed"]) / "app_cache.sqlite"


def _thinktime_dir(cfg: dict) -> Path:
    # Prefer the bundled model (ships with the app) over the research output dir.
    assets = Path(__file__).resolve().parents[1] / "assets" / "thinktime"
    if (assets / "expected_thinktime_model.joblib").exists():
        return assets
    return Path(cfg["paths"]["processed"]) / "thinktime"


# Bullet has its own think-time mapping and its own feature columns (Maia
# dispersion as the complexity signal, plus normalized clock, side, increment).
_BULLET_CAT = ["game_phase", "rating_band", "side", "increment"]
_BULLET_NUM = ["maia_dispersion", "norm_clock_remaining", "move_number"]


def _bullet_thinktime_dir(cfg: dict) -> Path:
    return _thinktime_dir(cfg) / "bullet"


def _bullet_expected(model, frame) -> list[float]:
    """Expected bullet think-time in seconds for each row (inverse of the log
    target the model was fit on)."""
    import numpy as np
    pred = model.predict(frame[_BULLET_CAT + _BULLET_NUM])
    return np.expm1(pred).clip(min=0.0)


def _stockfish_path(cfg: dict) -> str:
    # The launcher sets STOCKFISH_PATH to the fetched engine; fall back to config.
    return os.environ.get("STOCKFISH_PATH") or cfg["stockfish_path"]


def _maia_model_regime(regime: str | None) -> str:
    """Maia ships a blitz and a rapid model; map any speed to the nearest one so
    the human-difficulty signal is available for every game, not just blitz and
    rapid. Only the model choice is mapped; nothing else uses this."""
    return "rapid" if regime in ("rapid", "classical") else "blitz"


def band_label(elo: int) -> str:
    for lo, hi in pairwise(BANDS):
        if lo <= elo < hi:
            return f"[{lo}, {hi})"
    return "[1500, 1700)"


def difficulty_bucket(dispersion: float | None, c: ClassifyConfig) -> str | None:
    """Plain-language read of Maia dispersion, using the same lo/hi the gate uses."""
    if dispersion is None:
        return None
    if dispersion < c.disp_lo:
        return "routine"
    if dispersion < (c.disp_lo + c.disp_hi) / 2:
        return "tricky"
    if dispersion < c.disp_hi:
        return "hard"
    return "very hard"


def _parse_tc(tc: str) -> tuple[int, int]:
    """Base and increment seconds from a TimeControl header, lenient about odd
    or missing values (they only matter when clocks are present anyway)."""
    base, sep, inc = tc.strip().partition("+")
    try:
        b = int(base)
    except ValueError:
        b = 0
    try:
        i = int(inc) if sep else 0
    except ValueError:
        i = 0
    return b, i


@dataclass
class MoveResult:
    ply: int
    move_number: int
    side: str
    san: str
    uci: str
    fen_before: str
    phase: str
    in_book: bool
    # quality
    win_before: float
    win_after: float
    wpl: float
    eval_white: int  # eval after the move, White POV
    sac_cp: float  # material given up and not won back, mover POV (for the Brilliant rule)
    # clocks (None when the game has none)
    clock_before_s: float | None
    clock_after_s: float | None  # clock remaining after the move, straight from [%clk]
    time_spent_s: float | None
    premove_status: str  # premove, sub_floor_ambiguous, genuine, or unknown
    misclick_suspect: bool  # big Win% drop at floor time, likely a slip not a decision
    # under-pressure verdict for time-aware games: found_under_pressure,
    # not_pressured, insufficient_evidence, or None when it does not apply
    under_pressure: str | None
    expected_think_s: float | None
    residual_s: float | None
    # difficulty and pressure (None when time-aware is unavailable or in book)
    maia_entropy: float | None
    difficulty: str | None
    gate: float | None
    pressure: float | None
    skill: float | None
    # labels
    baseline_label: str
    time_aware_label: str | None
    upgraded: bool


@dataclass
class AnalysisResult:
    white: str
    black: str
    white_elo: int | None
    black_elo: int | None
    result: str
    opening: str
    site: str
    regime: str | None
    time_control: str
    final_fen: str  # position after the last move, so the board can show every ply
    time_aware_available: bool
    time_aware_note: str
    book_note: str | None  # set when opening theory could not be fully checked online
    moves: list[dict]
    summary: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _sac_series(moves, post_material: list[float]) -> list[float]:
    """Material given up per move, mover POV, that is NOT won back within the
    mover's next two turns.

    `post_material` is material AFTER each move (see caller), so a plain trade
    reads as even rather than a momentary deficit mid-exchange. We take the BEST
    material the mover recovers to over the next two of its own turns, so a piece
    won straight back, or a delayed recapture, does not read as a sacrifice. Only
    material that stays given up counts. Delayed recaptures were a known
    false-positive at Checkpoint A; measuring after the move is what fixes it.
    """
    sac = [0.0] * len(moves)
    for side in ("white", "black"):
        idx = [i for i, m in enumerate(moves) if m.side == side]
        mat = [post_material[k] for k in idx]
        for j, i in enumerate(idx):
            future = mat[j + 1 : j + 3]  # the mover's next two turns
            settled = max(future) if future else mat[j]
            sac[i] = max(0.0, mat[j] - settled)
    return sac


def analyze(report: GameReport, cfg: dict, *, option: str = "B",
            workers: int | None = None, progress=None) -> AnalysisResult:
    """Review one accepted game. `progress(stage, done, total)` reports the slow
    stages so a UI can show a bar. `option` picks the primary Option B upgrade
    rule or the Option A threshold sanity check."""
    if not report.accepted:
        raise ValueError(f"cannot analyze a rejected game: {report.reject_reason}")

    moves = report.moves
    n = len(moves)
    c = ClassifyConfig.from_config(cfg)
    workers = _service_workers(cfg) if workers is None else workers
    sf_path = _stockfish_path(cfg)

    def stage(name):
        return (lambda done, total: progress(name, done, total)) if progress else None

    if n == 0:
        return AnalysisResult(
            white=report.white, black=report.black, white_elo=report.white_elo,
            black_elo=report.black_elo, result=report.result, opening=report.opening,
            site=report.site, regime=report.regime, time_control=report.time_control,
            final_fen="", time_aware_available=False,
            time_aware_note="This game has no moves to review.", book_note=None,
            moves=[], summary={"n_upgrades": 0, "baseline_counts": {}},
        )

    # --- terminal position (for the last move's eval when the PGN has none) ---
    last = moves[-1]
    tb = chess.Board(last.fen_before)
    tb.push_uci(last.uci)
    terminal_fen = tb.fen()
    terminal_gameover = tb.is_game_over()
    terminal_checkmate = tb.is_checkmate()

    report_evals = [m.eval_cp_white for m in moves]
    engine_fens = [m.fen_before for m in moves]
    if report_evals[-1] is None and not terminal_gameover:
        engine_fens.append(terminal_fen)

    feats = engine.analyze_fens(engine_fens, cache := FenCache(_cache_path(cfg)),
                                sf_path, workers, progress=stage("engine"))
    try:
        # --- opening book: real theory from the bundled offline book ---------
        # A move is Book only if the book has it for that exact position, at any
        # move number. No network. If the book file is missing we do NOT guess
        # from ply count; those moves get their normal eval label and the report
        # says the book could not be loaded.
        reader = book.open_book(cfg)
        try:
            lookup = ((lambda fen: book.book_moves(reader, fen)) if reader
                      else (lambda fen: None))
            in_book, book_lookups, book_ok = book.detect_book(
                [m.fen_before for m in moves], [m.uci for m in moves], lookup)
        finally:
            if reader is not None:
                reader.close()
        book_note = None if book_ok else (
            "The opening book could not be loaded, so moves were classified by "
            "evaluation only.")

        # --- evals: use [%eval] where present, else the engine read ----------
        def resulting_eval_white(i: int) -> int:
            if report_evals[i] is not None:
                return report_evals[i]
            if i + 1 < n:
                return feats[moves[i + 1].fen_before]["eval_white"]
            if terminal_gameover:
                if terminal_checkmate:
                    return -MATE_CP if tb.turn == chess.WHITE else MATE_CP
                return 0
            return feats[terminal_fen]["eval_white"]

        eval_after_white = [resulting_eval_white(i) for i in range(n)]
        eval_before_white = [0 if i == 0 else eval_after_white[i - 1] for i in range(n)]

        entropy = [feats[m.fen_before]["entropy"] for m in moves]
        eval_gap = [feats[m.fen_before]["eval_gap"] for m in moves]
        # Material AFTER each move (mover POV), so a trade reads as even instead of
        # a momentary deficit while the mover is about to recapture. The sacrifice
        # detector needs this to tell a real sac from an ordinary exchange.
        def _post_fen(i: int) -> str:
            return moves[i + 1].fen_before if i + 1 < n else terminal_fen

        post_material = [material_cp(_post_fen(i), m.side == "white")
                         for i, m in enumerate(moves)]
        sac = _sac_series(moves, post_material)

        win_before, win_after, wpl = [], [], []
        for i, m in enumerate(moves):
            white = m.side == "white"
            bw = eval_before_white[i] if white else -eval_before_white[i]
            aw = eval_after_white[i] if white else -eval_after_white[i]
            wb, wa = win_pct_mate(bw), win_pct_mate(aw)
            win_before.append(wb)
            win_after.append(wa)
            wpl.append(max(0.0, wb - wa))

        # --- clocks ----------------------------------------------------------
        base, inc = _parse_tc(report.time_control)
        clock_before: list[float | None] = [None] * n
        time_spent: list[float | None] = [None] * n
        if report.has_clocks:
            prev = {"white": float(base), "black": float(base)}
            for i, m in enumerate(moves):
                if m.clock_s is None:
                    continue
                clock_before[i] = prev[m.side]
                time_spent[i] = max(0.0, prev[m.side] - m.clock_s + inc)
                prev[m.side] = m.clock_s

        # --- premove / sub-floor detection (the under-pressure guardrail) ------
        # A premove was not found at the board, so it can never earn a pressure
        # upgrade. move_time keeps its sign (an increment clock can rise on a
        # premove); we only trust it when the clock has sub-second precision.
        subsecond = premove.has_subsecond_clocks([m.clock_s for m in moves])
        premove_status = [premove.UNKNOWN] * n
        for i, m in enumerate(moves):
            if clock_before[i] is None or m.clock_s is None:
                continue
            move_time = clock_before[i] + inc - m.clock_s
            premove_status[i] = premove.classify(move_time, m.move_number == 1, subsecond)

        # --- rating bands (actual rating, fixed fallback when missing) -------
        def elo_of(m):
            e = report.white_elo if m.side == "white" else report.black_elo
            return e if e is not None else DEFAULT_ELO

        band = [band_label(elo_of(m)) for m in moves]
        band_elo = [maia.band_to_elo(b) for b in band]

        # --- Maia human-difficulty. Runs whenever Maia is installed, not only for
        # time-aware games, because the Brilliant rule needs the difficulty signal
        # too. The model is picked by regime (mapped to blitz/rapid); the reading
        # is at the player's own rating band, so difficulty is rating-relative. ---
        maia_ready = maia_client.available()
        maia_regime = _maia_model_regime(report.regime)
        maia_ent: list[float | None] = [None] * n
        if maia_ready:
            tasks = [(moves[i].fen_before, maia_regime, band_elo[i])
                     for i in range(n) if not in_book[i]]
            disp = maia_client.dispersion(tasks, cache, workers=1, progress=stage("maia"))
            for i in range(n):
                if not in_book[i]:
                    maia_ent[i] = disp.get((moves[i].fen_before, maia_regime, band_elo[i]))

        # --- time-aware pressure ---------------------------------------------
        # Bullet uses its own mapping and normalized pressure; blitz and rapid use
        # the original mapping and absolute-clock pressure. Both need clocks and
        # Maia, and the right model on disk.
        is_bullet = report.regime == "bullet"
        model_dir = _bullet_thinktime_dir(cfg) if is_bullet else _thinktime_dir(cfg)
        time_aware = report.time_aware_available and maia_ready \
            and (model_dir / "expected_thinktime_model.joblib").exists()
        note = report.capability_note
        if report.time_aware_available and not time_aware:
            note = ("Clocks are present, but the difficulty model is not installed, "
                    "so only the baseline review is shown. Run the install script "
                    "to enable the time-aware review.")
        elif is_bullet and time_aware:
            # State the honest bullet asymmetry: tenths (chess.com) are reliable,
            # whole seconds (public Lichess) cannot separate a premove from a fast
            # find, so those moves are never credited as finds under pressure.
            note = ("This is a bullet game with tenth-of-a-second clocks, so the "
                    "time-aware review is reliable and premoves are excluded."
                    if subsecond else
                    "This is a bullet game with whole-second clocks, so the "
                    "time-aware review is limited: a sub-second move cannot be told "
                    "apart from a premove, and those are not credited as finds "
                    "under pressure.")

        residual: list[float | None] = [None] * n
        expected: list[float | None] = [None] * n
        if time_aware:
            model = tt.load_mapping(model_dir)
            if is_bullet:
                frame = pd.DataFrame({
                    "maia_dispersion": [e if e is not None else 0.0 for e in maia_ent],
                    "norm_clock_remaining": [(cb / base) if (cb is not None and base) else 1.0
                                             for cb in clock_before],
                    "move_number": [float(m.move_number) for m in moves],
                    "game_phase": [m.phase for m in moves],
                    "rating_band": band,
                    "side": [m.side for m in moves],
                    "increment": ["yes" if inc else "no"] * n,
                })
                exp = _bullet_expected(model, frame)
            else:
                frame = pd.DataFrame({
                    "complexity": entropy,
                    "move_number": [float(m.move_number) for m in moves],
                    "log_clock_before": [math.log1p(max(0.0, cb or 0.0)) for cb in clock_before],
                    "game_phase": [m.phase for m in moves],
                    "regime": [report.regime] * n,
                    "rating_band": band,
                })
                exp = tt.expected_thinktime(model, frame)
            for i in range(n):
                expected[i] = float(exp[i])
                if time_spent[i] is not None:
                    residual[i] = time_spent[i] - float(exp[i])

        # --- guardrails: mouse-slip and flag artifacts -----------------------
        # A big Win% drop at floor time reads as a slip, not a considered move.
        # A game lost on time has an unreliable final move (lag / flag), so it is
        # not assessed for pressure either.
        misclick = [wpl[i] >= c.misclick_wpl_min
                    and premove_status[i] != premove.GENUINE for i in range(n)]
        timed_out = "time" in (report.termination or "").lower()

        # --- labels ----------------------------------------------------------
        results: list[MoveResult] = []
        for i, m in enumerate(moves):
            brilliant = is_brilliant(wpl[i], sac[i], win_before[i], win_after[i],
                                     maia_ent[i], c)
            great = is_great(wpl[i], eval_gap[i], c)
            base_label = baseline_label(wpl[i], in_book[i], brilliant, great, c)

            gate = pressure = skill = None
            ta_label = None
            under_pressure = None
            if time_aware:
                # A move cannot be assessed for pressure if it is theory, unscored,
                # a premove/sub-floor move, a suspected slip, or a flag artifact.
                blocked = (in_book[i] or maia_ent[i] is None or residual[i] is None
                           or not premove.can_be_under_pressure(premove_status[i])
                           or misclick[i] or (timed_out and i == n - 1))
                if blocked:
                    ta_label = base_label
                    if not in_book[i] and maia_ent[i] is not None and residual[i] is not None:
                        # scored, but the timing cannot be trusted: say so honestly
                        under_pressure = "insufficient_evidence"
                else:
                    gate = complexity_gate(maia_ent[i], c)
                    if is_bullet:
                        pressure = pressure_modifier_bullet(residual[i], clock_before[i], base, c)
                    else:
                        pressure = pressure_modifier(residual[i], clock_before[i], c)
                    skill = skill_of_move(wpl[i], gate, pressure, c)
                    if option == "A":
                        ta_label = option_a_label(base_label, wpl[i], maia_ent[i], pressure, c)
                    else:
                        ta_label = option_b_label(base_label, skill, brilliant, wpl[i], c)
                    under_pressure = ("found_under_pressure" if ta_label != base_label
                                      else "not_pressured")

            results.append(MoveResult(
                ply=m.ply, move_number=m.move_number, side=m.side, san=m.san,
                uci=m.uci, fen_before=m.fen_before, phase=m.phase, in_book=in_book[i],
                win_before=round(win_before[i], 2), win_after=round(win_after[i], 2),
                wpl=round(wpl[i], 2), eval_white=int(eval_after_white[i]),
                sac_cp=round(sac[i], 1),
                clock_before_s=clock_before[i], clock_after_s=m.clock_s,
                time_spent_s=time_spent[i], premove_status=premove_status[i],
                misclick_suspect=misclick[i], under_pressure=under_pressure,
                expected_think_s=round(expected[i], 1) if expected[i] is not None else None,
                residual_s=round(residual[i], 1) if residual[i] is not None else None,
                maia_entropy=round(maia_ent[i], 3) if maia_ent[i] is not None else None,
                difficulty=difficulty_bucket(maia_ent[i], c),
                gate=round(gate, 3) if gate is not None else None,
                pressure=round(pressure, 3) if pressure is not None else None,
                skill=round(skill, 3) if skill is not None else None,
                baseline_label=base_label, time_aware_label=ta_label,
                upgraded=bool(ta_label and ta_label != base_label),
            ))
    finally:
        cache.close()

    summary = _summarize(results, time_aware)
    summary["book_lookups"] = book_lookups
    summary["book_checked"] = book_ok
    return AnalysisResult(
        white=report.white, black=report.black, white_elo=report.white_elo,
        black_elo=report.black_elo, result=report.result, opening=report.opening,
        site=report.site, regime=report.regime, time_control=report.time_control,
        final_fen=terminal_fen, time_aware_available=time_aware, time_aware_note=note,
        book_note=book_note, moves=[asdict(r) for r in results], summary=summary,
    )


def _summarize(results: list[MoveResult], time_aware: bool) -> dict:
    baseline_counts = Counter(r.baseline_label for r in results)
    upgrades = [r for r in results if r.upgraded]
    # Premove / sub-floor rates, over moves that carried a clock reading.
    clocked = [r for r in results if r.premove_status != premove.UNKNOWN]
    premove_counts = Counter(r.premove_status for r in clocked)
    summary = {
        "n_moves": len(results),
        "baseline_counts": dict(baseline_counts),
        "premove_counts": dict(premove_counts),
        "time_aware_counts": (dict(Counter(r.time_aware_label for r in results if r.time_aware_label))
                              if time_aware else None),
        "n_upgrades": len(upgrades),
        "upgrades": [{
            "ply": r.ply, "move_number": r.move_number, "side": r.side, "san": r.san,
            "baseline": r.baseline_label, "time_aware": r.time_aware_label,
            "wpl": r.wpl, "maia_entropy": r.maia_entropy, "difficulty": r.difficulty,
            "time_spent_s": r.time_spent_s, "expected_think_s": r.expected_think_s,
            "clock_before_s": r.clock_before_s, "fen_before": r.fen_before,
        } for r in upgrades],
    }
    return summary
