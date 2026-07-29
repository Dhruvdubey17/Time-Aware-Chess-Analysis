"""Phase 1: turn any pasted or uploaded PGN into clean, analyzable moves, and
report honestly what it contains.

One PGN string can hold several games. We split them, and for each game we say
plainly: which site made it, is there a clock on the moves, is there an engine
evaluation, what is the time control, how many moves. Standard-chess games get a
normalized per-move list that later phases consume. Variants and non-standard
starting positions are rejected with a clear message, because the classifier is
built for standard chess only.

No engine here. This is pure parsing and detection. Missing evaluations are not
computed yet (Phase 2 does that); we only report that they are missing.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field

import chess
import chess.pgn

from chess_strength.features import game_phase
from chess_strength.stream_filter import classify_time_control

# python-chess offsets a mate score by its distance, same convention as
# parse_moves, so anything near this magnitude is a mate not a normal eval.
MATE_CP = 10000

# First plies are opening theory. Matches config/default.yaml book_plies.
DEFAULT_BOOK_PLIES = 12

# The time-aware mapping and the Maia models were fitted on blitz and rapid
# only. Other speeds still get a baseline review, never a faked time signal.
TIME_AWARE_REGIMES = {"blitz", "rapid"}

_STANDARD_VARIANTS = {"", "standard", "chess", "normal"}


@dataclass
class Move:
    """One ply, normalized. Clock and eval are None when the PGN does not carry
    them. eval_cp_white is the White point of view, exactly as `[%eval]` stores
    it, so later phases flip it the same way for either side."""

    ply: int
    move_number: int
    side: str  # "white" or "black"
    san: str
    uci: str
    fen_before: str
    clock_s: float | None
    eval_cp_white: int | None
    phase: str  # opening, middlegame, or endgame
    in_book: bool


@dataclass
class GameReport:
    """Everything we can honestly say about one game before any engine runs."""

    index: int  # position in the file, 0-based
    accepted: bool
    reject_reason: str | None

    # Headers worth showing the player.
    white: str
    black: str
    white_elo: int | None
    black_elo: int | None
    result: str
    termination: str
    opening: str

    # Detection.
    site: str  # "lichess", "chess.com", or "unknown"
    time_control: str
    regime: str | None
    has_clocks: bool
    has_evals: bool
    n_plies: int
    n_moves: int

    # Capability, in plain language for the UI.
    time_aware_available: bool
    capability_note: str

    moves: list[Move] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain dict for JSON, nested moves included."""
        return asdict(self)

    def summary(self) -> dict:
        """The capability report without the per-move payload."""
        d = asdict(self)
        d.pop("moves")
        return d


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _detect_site(headers: chess.pgn.Headers) -> str:
    blob = f"{headers.get('Site', '')} {headers.get('Link', '')}".lower()
    if "lichess" in blob:
        return "lichess"
    if "chess.com" in blob or "chesscom" in blob:
        return "chess.com"
    return "unknown"


def _reject_reason(game: chess.pgn.Game) -> str | None:
    """Why we cannot review this game, or None if it is standard chess.

    Two things put a game out of scope: a non-standard variant, and a game that
    does not start from the normal chess position (custom FEN, puzzle, study).
    """
    variant = game.headers.get("Variant", "").strip().lower()
    if variant not in _STANDARD_VARIANTS:
        name = game.headers.get("Variant", "").strip()
        return f"This is a {name} game. The review works on standard chess only."

    fen = game.headers.get("FEN", "").strip()
    if fen and fen != chess.STARTING_FEN:
        return ("This game does not start from the normal chess position, so it "
                "cannot be reviewed.")
    return None


def _parse_moves(game: chess.pgn.Game, book_plies: int) -> tuple[list[Move], bool, bool]:
    """Walk the mainline into normalized rows. Also reports whether any clock or
    any eval was seen, which is what the capability report turns on."""
    board = game.board()
    moves: list[Move] = []
    has_clocks = False
    has_evals = False

    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        side = "white" if board.turn == chess.WHITE else "black"
        fen_before = board.fen()
        san = board.san(move)  # must read the SAN before the move is played

        clock = node.clock()
        score = node.eval()
        eval_cp_white = score.white().score(mate_score=MATE_CP) if score is not None else None

        has_clocks = has_clocks or clock is not None
        has_evals = has_evals or eval_cp_white is not None

        moves.append(
            Move(
                ply=ply,
                move_number=(ply + 1) // 2,
                side=side,
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                clock_s=clock,
                eval_cp_white=eval_cp_white,
                phase=game_phase(fen_before, ply, book_plies),
                in_book=ply <= book_plies,
            )
        )
        board.push(move)

    return moves, has_clocks, has_evals


def _capability(has_clocks: bool, has_evals: bool, regime: str | None,
                n_plies: int, had_errors: bool) -> tuple[bool, str]:
    """Plain-language verdict on what kind of review this game supports."""
    if n_plies == 0:
        return False, "This game has no moves to review."

    time_aware = has_clocks and regime in TIME_AWARE_REGIMES

    if time_aware:
        lead = "Clocks are present, so the time-aware review is available."
    elif not has_clocks:
        lead = ("There are no move clocks in this PGN, so time pressure cannot be "
                "measured. Only the baseline review is available.")
    else:
        speed = regime or "an unusual time control"
        lead = (f"This is {speed}. The time-aware review is calibrated for blitz "
                f"and rapid, so only the baseline review is shown.")

    evals = (" Evaluations are included." if has_evals
             else " Evaluations are missing and will be computed on your machine, "
                  "which takes a little longer.")

    warn = (" Some moves could not be read; the review covers the moves that parsed."
            if had_errors else "")

    return time_aware, lead + evals + warn


def build_report(game: chess.pgn.Game, index: int,
                 book_plies: int = DEFAULT_BOOK_PLIES) -> GameReport:
    """One game to a full report. Rejected games carry the reason and no moves."""
    h = game.headers
    reject = _reject_reason(game)
    tc = h.get("TimeControl", "").strip()
    regime = classify_time_control(tc) if tc else None

    common = {
        "index": index,
        "white": h.get("White", ""),
        "black": h.get("Black", ""),
        "white_elo": _int_or_none(h.get("WhiteElo")),
        "black_elo": _int_or_none(h.get("BlackElo")),
        "result": h.get("Result", "*"),
        "termination": h.get("Termination", ""),
        "opening": h.get("Opening", ""),
        "site": _detect_site(h),
        "time_control": tc,
        "regime": regime,
    }

    if reject is not None:
        return GameReport(
            accepted=False, reject_reason=reject,
            has_clocks=False, has_evals=False, n_plies=0, n_moves=0,
            time_aware_available=False, capability_note=reject,
            moves=[], **common,
        )

    moves, has_clocks, has_evals = _parse_moves(game, book_plies)
    n_plies = len(moves)
    n_moves = moves[-1].move_number if moves else 0
    time_aware, note = _capability(has_clocks, has_evals, regime, n_plies,
                                   bool(game.errors))

    return GameReport(
        accepted=True, reject_reason=None,
        has_clocks=has_clocks, has_evals=has_evals, n_plies=n_plies, n_moves=n_moves,
        time_aware_available=time_aware, capability_note=note,
        moves=moves, **common,
    )


def parse_pgn(text: str, book_plies: int = DEFAULT_BOOK_PLIES) -> list[GameReport]:
    """Every game in a pasted PGN string, in order. Empty or unparseable input
    gives an empty list; the caller decides how to tell the user."""
    reports: list[GameReport] = []
    stream = io.StringIO(text)
    index = 0
    while True:
        # This is a trust boundary: the text is whatever the user pasted. Bad
        # moves are recorded on the game and handled downstream; only a truly
        # unreadable stream would raise, and then we just stop reading.
        try:
            game = chess.pgn.read_game(stream)
        except Exception:  # noqa: BLE001 - malformed paste, stop gracefully
            break
        if game is None:
            break
        reports.append(build_report(game, index, book_plies))
        index += 1
    return reports


def parse_pgn_file(path, book_plies: int = DEFAULT_BOOK_PLIES) -> list[GameReport]:
    """Same as parse_pgn but reads an uploaded `.pgn` file from disk."""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_pgn(text, book_plies)
