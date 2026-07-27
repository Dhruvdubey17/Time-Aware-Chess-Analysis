"""Phase 2: stream a Lichess monthly dump and keep only the games we can use.

A monthly standard file is a `.pgn.zst` that unpacks to 200+ GB, so we never
decompress it to disk. We read the compressed stream game by game and keep a
game only if it carries both a `[%clk]` and a `[%eval]` comment and its time
control is one of the regimes we care about. Matches are written to capped
`.pgn` shards so nothing downstream ever needs the giant original.

The filter works on any line source, so the same code runs on the tiny plain
`.pgn` fixture in tests and on a real `.pgn.zst` in production.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_TIME_CONTROL_RE = re.compile(r'\[TimeControl "([^"]+)"\]')
_SHARD_CAP_BYTES = 200 * 1024 * 1024


def classify_time_control(tc: str) -> str | None:
    """Map a PGN TimeControl to a Lichess speed regime.

    Lichess buckets by an estimated game length of base + 40*increment seconds,
    the same thresholds their site uses. `"-"` (correspondence / unlimited) and
    anything unparseable return None.
    """
    tc = tc.strip()
    if tc in ("-", "", "?"):
        return None
    base, _, inc = tc.partition("+")
    try:
        estimated = int(base) + 40 * int(inc or 0)
    except ValueError:
        return None
    if estimated < 30:
        return "ultrabullet"
    if estimated < 180:
        return "bullet"
    if estimated < 480:
        return "blitz"
    if estimated < 1500:
        return "rapid"
    return "classical"


def iter_games(lines: Iterable[str]) -> Iterator[str]:
    """Split a PGN line stream into raw game text blocks.

    Every Lichess game starts with an `[Event ...]` tag, so we cut on that. We
    keep the text as-is instead of fully parsing it here; Phase 3 does the real
    parse. Anything before the first game (there normally is nothing) is skipped.
    """
    buf: list[str] = []
    for line in lines:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            buf = [line]
        else:
            buf.append(line)
    if any(chunk.strip() for chunk in buf):
        yield "".join(buf)


def game_regime(game_text: str, regimes: Iterable[str]) -> str | None:
    """Return the game's regime if it is one we keep, else None."""
    m = _TIME_CONTROL_RE.search(game_text)
    if not m:
        return None
    regime = classify_time_control(m.group(1))
    return regime if regime in set(regimes) else None


def game_is_usable(game_text: str, regimes: Iterable[str]) -> str | None:
    """A game is usable if it has both tags and a wanted regime.

    Returns the matched regime (truthy) or None so callers can filter and label
    in one step.
    """
    if "%clk" not in game_text or "%eval" not in game_text:
        return None
    return game_regime(game_text, regimes)


def filter_games(
    lines: Iterable[str], regimes: Iterable[str]
) -> Iterator[tuple[str, str]]:
    """Yield (game_text, regime) for every usable game in the stream."""
    regimes = set(regimes)
    for game in iter_games(lines):
        regime = game_is_usable(game, regimes)
        if regime:
            yield game, regime


def _open_lines(path: Path) -> Iterator[str]:
    """Line iterator over a `.pgn` or a streamed `.pgn.zst`, never fully decompressed."""
    if path.suffix == ".zst":
        import zstandard

        fh = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        try:
            yield from text
        finally:
            text.close()
            fh.close()
    else:
        with path.open(encoding="utf-8") as fh:
            yield from fh


@dataclass
class FilterStats:
    games_seen: int = 0
    games_kept: int = 0
    shards_written: int = 0

    @property
    def keep_rate(self) -> float:
        return self.games_kept / self.games_seen if self.games_seen else 0.0


class _ShardWriter:
    """Write games to capped `.pgn` shards, rolling to a new file at the cap."""

    def __init__(self, out_dir: Path, cap_bytes: int):
        self.out_dir = out_dir
        self.cap_bytes = cap_bytes
        self.index = -1
        self.written = 0
        self.count = 0
        self._fh: io.TextIOBase | None = None
        out_dir.mkdir(parents=True, exist_ok=True)

    def _roll(self) -> None:
        if self._fh:
            self._fh.close()
        self.index += 1
        self.written = 0
        path = self.out_dir / f"shard_{self.index:03d}.pgn"
        self._fh = path.open("w", encoding="utf-8")

    def write(self, game_text: str) -> None:
        block = game_text.strip() + "\n\n"
        if self._fh is None or self.written + len(block) > self.cap_bytes:
            self._roll()
        assert self._fh is not None
        self._fh.write(block)
        self.written += len(block)
        self.count += 1

    def close(self) -> None:
        if self._fh:
            self._fh.close()


def run_filter(
    src_path: str | Path,
    out_dir: str | Path,
    regimes: Iterable[str],
    cap_bytes: int = _SHARD_CAP_BYTES,
) -> FilterStats:
    """Filter one source file into `.pgn` shards and return counts.

    Streams the source, keeps usable games, writes them to shards under out_dir.
    Works the same on the fixture and on a real monthly `.pgn.zst`.
    """
    regimes = set(regimes)
    stats = FilterStats()
    writer = _ShardWriter(Path(out_dir), cap_bytes)
    try:
        for game in iter_games(_open_lines(Path(src_path))):
            stats.games_seen += 1
            if game_is_usable(game, regimes):
                writer.write(game)
                stats.games_kept += 1
    finally:
        writer.close()
    stats.shards_written = writer.index + 1 if writer.count else 0
    return stats
